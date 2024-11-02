import json
import os
import shutil
import subprocess

# Load prompts from json file
with open("data/prompt_img_pairs.json", "r") as f:
    prompt_data = json.load(f)

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Run main.py for each prompt
for prompt_key, data in prompt_data.items():
    prompt = data["prompt"]
    edit_prompt = data["edit_prompt"]
    src_img_path = data["img_path"].replace('$HOME/data/imgs/', './sds_outputs/')
    
    # Create output subdirectory for this prompt
    save_dir = os.path.join("pds_outputs", prompt_key)
    os.makedirs(save_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "main.py",
        "--prompt", prompt,
        "--edit_prompt", edit_prompt,
        "--src_img_path", src_img_path,
        "--loss_type", "pds", 
        "--guidance_scale", "25.0",
        "--save_dir", save_dir
    ]
    
    print(f"\nProcessing prompt: {prompt}")
    subprocess.run(cmd)
    shutil.copy(save_dir + f'/475.png', save_dir + f'/../{prompt_key}.png')
