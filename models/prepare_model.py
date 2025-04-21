import torch
import torch.nn as nn

def get_dino_vit_model(num_classes=100, freeze_backbone=True):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    for param in model.parameters():
        param.requires_grad = False
    
    output_shape = num_classes
    model.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=384,
                    out_features=output_shape,
                    bias=True))
    # # Save the embedding dim
    # embed_dim = 384

    # # Replace the head with a new classifier head
    # model.head = nn.Linear(embed_dim, num_classes)

    # # Freeze the backbone (everything except the new head)
    # if freeze_backbone:
    #     for name, param in model.named_parameters():
    #         if not name.startswith("head."):
    #             param.requires_grad = False

    return model
