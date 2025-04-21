import torch
import torch.nn as nn

def get_dino_vit_model(num_classes=100, freeze_backbone=True):
    """
    Load pretrained DINO ViT-S/16 and adapt it for CIFAR-100 classification.
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    # Freeze all layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # DINO removes classification head → we add a new one
    # The output embedding dim for ViT-S/16 is 384
    model.head = nn.Linear(384, num_classes)

    return model
