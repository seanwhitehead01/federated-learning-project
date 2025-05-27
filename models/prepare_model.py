import torch
import torch.nn as nn

def get_dino_vits16_model(device, num_classes=100):
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
    for param in model.backbone.parameters():
        param.requires_grad = False

def freeze_head(model):
    for param in model.head.parameters():
        param.requires_grad = False

def unfreeze_head(model):
    for param in model.head.parameters():
        param.requires_grad = True

def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True