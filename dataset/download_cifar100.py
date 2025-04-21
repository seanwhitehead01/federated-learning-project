import os
import torch
from torchvision import datasets, transforms

def download_and_save_cifar100(data_dir="dataset", save_name="cifar100"):
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, f"{save_name}_train.pt")
    test_path = os.path.join(data_dir, f"{save_name}_test.pt")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("✅ CIFAR-100 already downloaded and saved.")
        return

    transform = transforms.Compose([transforms.ToTensor()])

    print("📥 Downloading CIFAR-100 training set...")
    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)

    print("📥 Downloading CIFAR-100 test set...")
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    print(f"💾 Saving training set to {train_path}")
    torch.save(trainset, train_path)

    print(f"💾 Saving test set to {test_path}")
    torch.save(testset, test_path)

    print("✅ Done! Dataset available in:", os.path.abspath(data_dir))

if __name__ == "__main__":
    download_and_save_cifar100()
