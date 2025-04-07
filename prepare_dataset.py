import os
import shutil
from torchvision import datasets
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from PIL import Image

def setup_directories():
    # Create directories if they don't exist
    dirs = ['data/train/images', 'data/val/images', 'data/test/images']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def copy_example_images():
    # Copy example images to test directory
    example_dir = './example_pics'
    if os.path.exists(example_dir):
        for img_name in os.listdir(example_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                src = os.path.join(example_dir, img_name)
                dst = os.path.join('data/test/images', img_name)
                shutil.copy2(src, dst)
        print(f"Copied example images to test directory")

def download_and_prepare_cifar():
    # Download CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Download CIFAR-10 training set
    trainset = datasets.CIFAR10(
        root='./data/cifar',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download CIFAR-10 test set
    testset = datasets.CIFAR10(
        root='./data/cifar',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)
    
    # Set counts for each split
    train_count = 1000  # Adjust these numbers as needed
    val_count = 200
    
    # Process and save training images
    count = 0
    for img, _ in train_loader:
        if count < train_count:
            save_dir = 'data/train/images'
            img_path = os.path.join(save_dir, f'cifar_train_{count:05d}.jpg')
            transforms.ToPILImage()(img.squeeze()).save(img_path)
            count += 1
            
            if count % 100 == 0:
                print(f'Processed {count} training images')
        else:
            break
    
    # Process and save validation images
    count = 0
    for img, _ in test_loader:
        if count < val_count:
            save_dir = 'data/val/images'
            img_path = os.path.join(save_dir, f'cifar_val_{count:05d}.jpg')
            transforms.ToPILImage()(img.squeeze()).save(img_path)
            count += 1
            
            if count % 50 == 0:
                print(f'Processed {count} validation images')
        else:
            break

if __name__ == '__main__':
    print("Setting up directories...")
    setup_directories()
    
    print("Copying example images...")
    copy_example_images()
    
    print("Downloading and preparing CIFAR-10 dataset...")
    download_and_prepare_cifar()
    
    print("Dataset preparation complete!")
    
    # Print statistics
    train_images = len(os.listdir('data/train/images'))
    val_images = len(os.listdir('data/val/images'))
    test_images = len(os.listdir('data/test/images'))
    
    print(f"\nDataset statistics:")
    print(f"Training images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Test images: {test_images}") 