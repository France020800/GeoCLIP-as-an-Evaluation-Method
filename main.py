import torch
import logging
import argparse
from prompt_generator import generate_dictionary
from image_generator import generate_images
from datasets import ImageDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from NaturalImageDataset import NaturalImageDataset
import os
import eval
from glob import glob
from PIL import Image
from geoclip import GeoCLIP

logging.basicConfig(
    filename='geoclip.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def main():
    logging.info('Starting GeoCLIP experiments')
    parser = argparse.ArgumentParser(description="GeoCLIP main entry point")
    parser.add_argument('--data_dir', type=str, help='Path to input file or data')
    parser.add_argument('--dataset_size', type=int, default=100, help='Size of the dataset to generate')
    parser.add_argument('--images_class', type=str, default='city', help='Class of images to generate')
    args = parser.parse_args()
    logging.info(f'Configuration: {args}')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.data_dir is not None:
        # Load existing dataset
        pass
    else:
        prompt = (
            f"Generate a valid Python dictionary (not code block) with exactly {args.dataset_size} entries, where the keys are prompts to generate {args.images_class} images, "
            "and the value is the GPS location of the city. Only output the dictionary."
        )
        print('Generating prompts...')
        image_dict = generate_dictionary(prompt)
        print('Prompts generated successfully!')
        prompts = list(image_dict.keys())
        logging.info('Generated prompts:')
        for prompt in prompts:
            logging.info(prompt)

        logging.info('Starting generating images...')
        generate_images(prompts, device=device)
        logging.info('Images generated successfully!')

        image_dataset = ImageDataset.ImageDataset('images', image_dict)
        image_loader = DataLoader(image_dataset, batch_size=8, shuffle=True)
        logging.info('Image dataset created successfully!')

    model = GeoCLIP().to("cuda:1")
    print("===========================")
    print("GeoCLIP has been loaded! 🎉")
    print("===========================")
    logging.info('GoeCLIP model loaded successfully!')




if __name__ == '__main__':
    main()