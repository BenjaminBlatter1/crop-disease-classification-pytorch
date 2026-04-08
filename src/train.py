from config import Config
from dataset import CustomImageDataset

from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    print("Day 2: Testing DataLoader...")

    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.ToTensor(),
    ])

    train_ds = CustomImageDataset(Config.train_dir, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )

    # Test one batch
    for images, labels in train_loader:
        print("Batch loaded:")
        print("Images:", images.shape)
        print("Labels:", labels.shape)
        break


if __name__ == "__main__":
    main()
