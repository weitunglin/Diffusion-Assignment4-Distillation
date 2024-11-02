from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        timestep = torch.randint(
            low=self.min_step,
            high=self.max_step + 1, 
            size=(1,),
            dtype=torch.long,
            device=self.device
        )

        with torch.no_grad():
            random_noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(
                original_samples=latents,
                noise=random_noise,
                timesteps=timestep
            )
            
            model_input = torch.cat([noisy_latents] * 2)
            predicted_noise = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings
            ).sample

        uncond_pred, cond_pred = predicted_noise.chunk(2)
        guided_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        timestep_weight = 1 - self.alphas[timestep]
        noise_diff = guided_pred - random_noise
        gradient = timestep_weight * noise_diff
        gradient = torch.nan_to_num(gradient)

        latents.backward(gradient=gradient, retain_graph=True)
        
        return torch.tensor([0.0], device=self.device)
    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        batch_size = src_latents.shape[0]
        self.scheduler.set_timesteps(self.num_train_timesteps)
        timesteps = reversed(self.scheduler.timesteps)
        max_step = max(self.max_step, self.min_step + 1)
        
        idx = torch.randint(
            self.min_step, max_step, 
            [batch_size], 
            dtype=torch.long, 
            device="cpu"
        )
        
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        beta_t = self.scheduler.betas[t].to(self.device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(self.device) 
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(self.device)
        sigma_t = torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t)

        noise = torch.randn_like(tgt_latents)
        noise_t_prev = torch.randn_like(tgt_latents)

        zts = {}
        latents_and_embeddings = [
            (src_latents, src_text_embedding, "src"),
            (tgt_latents, tgt_text_embedding, "tgt")
        ]
        
        for latent, text_embed, name in latents_and_embeddings:
            noisy_latents = self.scheduler.add_noise(latent, noise, t)
            
            model_input = torch.cat([noisy_latents] * 2, dim=0)
            t_input = torch.cat([t] * 2).to(self.device)
            noise_pred = self.unet(
                model_input,
                t_input, 
                encoder_hidden_states=text_embed
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            x_t_prev = self.scheduler.add_noise(latent, noise_t_prev, t_prev)
            
            beta_t = self.scheduler.betas[t].to(self.device)
            alpha_t = self.scheduler.alphas[t].to(self.device)
            alpha_bar_t = self.scheduler.alphas_cumprod[t].to(self.device)
            alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(self.device)

            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
            sqrt_alpha = torch.sqrt(alpha_bar_t)
            pred_x0 = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

            c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
            c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            mu = c0 * pred_x0 + c1 * noisy_latents
            
            zts[name] = (x_t_prev - mu) / sigma_t

        grad = zts["tgt"] - zts["src"]
        grad = torch.nan_to_num(grad)
        target = (tgt_latents - grad).detach()
        loss = 0.5 * torch.nn.functional.mse_loss(tgt_latents, target)
        
        return loss
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
