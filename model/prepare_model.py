import torch
import torch.nn as nn

def get_dino_vits16_model(device, num_classes=100):
    """
    Load the DINO ViT-S/16 model with a classification head for a specified number
    of classes.
    Args:
        device: The device to load the model onto (CPU or GPU).
        num_classes: Number of output classes for the classification head.
    Returns:
        nn.Module: The DINO ViT-S/16 model with a classification head.
    """
    # Load the DINO ViT-S/16 model from Facebook's repository
    base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    # Compose full model with additional classification head
    class DinoClassifier(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(backbone.num_features, num_classes)

        def forward(self, x):
            x = self.backbone(x)
            return self.head(x)

    model = DinoClassifier(base_model, num_classes)
    return model.to(device)

def freeze_backbone(model):
    """Freeze the backbone of the model so that its parameters are not updated during training.
    Args:
        model: The model whose backbone parameters are to be frozen.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

def freeze_head(model):
    """Freeze the head of the model so that its parameters are not updated during training.
    Args:
        model: The model whose head parameters are to be frozen.
    """
    for param in model.head.parameters():
        param.requires_grad = False

def unfreeze_head(model):
    """Unfreeze the head of the model so that its parameters can be updated during training.
    Args:
        model: The model whose head parameters are to be unfrozen.
    """
    for param in model.head.parameters():
        param.requires_grad = True

def unfreeze_backbone(model):
    """Unfreeze the backbone of the model so that its parameters can be updated during training.
    Args:
        model: The model whose backbone parameters are to be unfrozen.
    """
    for param in model.backbone.parameters():
        param.requires_grad = True