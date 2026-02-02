import torch
import torch.nn as nn
import torchvision.models as models

class TemporalCNN_GRU(nn.Module):
    """
    CNN + BiGRU temporal model
    Input: (B, T, C, H, W)
    """
    def __init__(self, num_classes=2, hidden_size=256, freeze_backbone=True):
        super().__init__()

        # CNN Backbone
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

                # Unfreeze last ResNet block for adaptation
            for p in backbone.layer4.parameters():
                p.requires_grad = True

        # Temporal Module
        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_dim = hidden_size * 2

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_temporal=False):
        B, T, C, H, W = x.shape

        # CNN per-frame
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)  # (B*T, 512, 1, 1)
        features = features.view(B, T, self.feature_dim)

        # Temporal modeling
        temporal_out, _ = self.temporal(features)  # (B, T, 2*hidden)

        # Temporal pooling (last frame)
        pooled = temporal_out.mean(dim=1)

        logits = self.classifier(pooled)

        if return_temporal:
            return logits, temporal_out

        return logits
    
    # In model.py, inside TemporalCNN_GRU
    def forward_single_frame(self, x):
        # x: (B, 3, H, W)
        B = x.shape[0]
        features = self.feature_extractor(x)  # (B, 512, 1, 1)
        features = features.view(B, self.feature_dim)
        logits = self.classifier(features)
        return logits
