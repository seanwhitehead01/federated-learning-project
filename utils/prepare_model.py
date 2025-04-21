import torch.nn as nn
import timm

# Load the pretrained DINO ViT-S/16 model
def get_dino_vit_model(num_classes=100, pretrained=True):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    
    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the last classification layer to match CIFAR-100's 100 classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    # Unfreeze the final classification layer
    for param in model.head.parameters():
        param.requires_grad = True

    return model