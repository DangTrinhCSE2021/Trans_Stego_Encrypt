import os
import random
import shutil
from torchvision import datasets
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader

def setup_directories():
    # Create directories if they don't exist
    dirs = ['data/train/images', 'data/val/images', 'data/test/images']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def download_and_split_data():
    # Download ImageNet validation set
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Download ImageNet validation set
    dataset = datasets.ImageNet(
        root='./data/imagenet',
        split='val',
        transform=transform,
        download=True
    )
    
    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Set counts for each split
    train_count = 1000  # Adjust these numbers as needed
    val_count = 200
    test_count = 100
    
    # Process and save images
    count = 0
    for img, _ in dataloader:
        if count < train_count:
            save_dir = 'data/train/images'
        elif count < train_count + val_count:
            save_dir = 'data/val/images'
        elif count < train_count + val_count + test_count:
            save_dir = 'data/test/images'
        else:
            break
            
        # Save the image
        img_path = os.path.join(save_dir, f'image_{count:05d}.jpg')
        transforms.ToPILImage()(img.squeeze()).save(img_path)
        count += 1
        
        if count % 100 == 0:
            print(f'Processed {count} images')

if __name__ == '__main__':
    setup_directories()
    download_and_split_data() 