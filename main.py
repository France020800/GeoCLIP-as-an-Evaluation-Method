import torch
import logging
import argparse
import json
import eval
import os

from prompt_generator import generate_dictionary
from image_generator import generate_images
from datasets.ImagePathDataset import ImagePathDataset
from torch.utils.data import DataLoader
from geoclip import GeoCLIP

logging.basicConfig(
    filename='geoclip.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def main():
    logging.info('**************************************')
    logging.info('**** Starting GeoCLIP experiments ****')
    logging.info('**************************************')
    parser = argparse.ArgumentParser(description="GeoCLIP main entry point")
    parser.add_argument('--data_dir', type=str, help='Path to input file or data')
    parser.add_argument('--dataset_size', type=int, default=100, help='Size of the dataset to generate')
    parser.add_argument('--images_class', type=str, default='city', help='Class of city to generate')
    args = parser.parse_args()
    logging.info(f'Configuration: {args}')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    if args.data_dir is not None:
        data_dir = args.data_dir
        logging.info(f'Using provided data directory: {data_dir}')

        image_dataset = ImagePathDataset(data_dir)
        image_loader = DataLoader(image_dataset, batch_size=8, shuffle=True)
        logging.info('Image dataset created successfully!')
    else:
        dataset_size = args.dataset_size
        images_class = args.images_class.replace(' ', '_')
        prompt = (
            f"Generate a valid Python dictionary (not code block) with exactly {dataset_size} entries, where the keys are prompts to generate famous {args.images_class}, "
            f"and the value is the tuple of floats GPS location of the {args.images_class}. Only output the dictionary."
        )
        print('Generating prompts...')
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        print(GEMINI_API_KEY)
        image_dict = generate_dictionary(prompt, API_KEY=GEMINI_API_KEY)
        print('Prompts generated successfully!')

        os.makedirs(f'datasets/{images_class}', exist_ok=True)
        with open(f'datasets/{images_class}/prompts.json', 'w') as fp:
            json.dump(image_dict, fp)

        prompts = list(image_dict.keys())
        logging.info('Generated prompts:')
        for prompt in prompts:
            logging.info(prompt)

        logging.info(f'Starting generating {images_class}...')
        generate_images(prompts, images_class, device=device)
        logging.info('Images generated successfully!')

        image_dataset = ImagePathDataset(f'datasets/{images_class}')
        image_loader = DataLoader(image_dataset, batch_size=8, shuffle=True)
        logging.info('Image dataset created successfully!')

    model = GeoCLIP().to(device)
    print("===========================")
    print("GeoCLIP has been loaded! 🎉")
    print("===========================")
    logging.info('GoeCLIP model loaded successfully!')

    results = eval.eval_images(image_loader, model, image_dataset=image_dataset)
    logging.info(json.dumps(results, indent=2))



if __name__ == '__main__':
    main()