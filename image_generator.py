from diffusers import DiffusionPipeline
import torch
import os
from tqdm import tqdm

def generate_images(prompts, image_class, device='cuda:1'):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to(device)

    for prompt in tqdm(prompts):
        dir = f'datasets/{image_class}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        print(f'Start generating promt: {prompt}')
        try:
            image = pipe(prompt=prompt).images[0]
            formatted_prompt = prompt.replace(' ', '_').replace("'", "").replace(',', '').replace('-', '')
            image.save(f"{dir}/{formatted_prompt}.png")
            del image
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")


if __name__ == '__main__':
    prompts = ['A cartoon style Florence landscape.']
    generate_images(prompts, 'test')