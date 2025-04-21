import torch
import torch.nn as nn

def get_dino_vit_model(num_classes=100, freeze_backbone=True):
    """
    Loads the pretrained DINO ViT-S/16 model from torch.hub and adapts it for CIFAR-100.

    Args:
        num_classes (int): number of output classes. Default is 100 (CIFAR-100).
        freeze_backbone (bool): if True, freezes all layers except the classification head.

    Returns:
        model (torch.nn.Module): the modified DINO ViT-S/16 model.
    """
    # Load the pretrained DINO ViT-S/16 model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    # Freeze the backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model
