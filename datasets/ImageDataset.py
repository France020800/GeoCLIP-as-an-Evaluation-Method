import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, prompt_to_gps):
        self.image_dir = image_dir
        self.prompt_to_gps = prompt_to_gps
        self.prompts = list(prompt_to_gps.keys())
        self.images = []
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        for prompt in self.prompts:
            formatted_prompt = prompt.replace(' ', '_').replace("'", "").replace(',', '').replace('-', '')
            image_path = os.path.join(image_dir, f"{formatted_prompt}.png")
            if os.path.exists(image_path):
                self.images.append((image_path, prompt_to_gps[prompt]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, gps = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(gps, dtype=torch.float)

if __name__ == '__main__':
    dataset = ImageDataset('images', dict)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)