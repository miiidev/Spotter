import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class TemporalCNN_GRU(nn.Module):
    def __init__(self, num_classes=2, hidden_size=512, freeze_backbone=True):
        super().__init__()

        # EfficientNet-B0 Backbone (better than ResNet18!)
        # Pretrained on ImageNet
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Remove the final classification layer
        # EfficientNet-B0 has 1280 features before the final FC layer
        self.feature_dim = 1280
        
        # Extract features (everything except the classifier)
        self.feature_extractor = nn.Sequential(
            self.backbone._conv_stem,
            self.backbone._bn0,
            self.backbone._swish,
            *self.backbone._blocks,
            self.backbone._conv_head,
            self.backbone._bn1,
            self.backbone._swish,
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )

        # Freeze/unfreeze strategy
        if freeze_backbone:
            # Freeze all backbone layers
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            
            # Unfreeze last few blocks for fine-tuning
            # EfficientNet has 16 blocks, unfreeze last 3
            for block in self.backbone._blocks[-3:]:
                for p in block.parameters():
                    p.requires_grad = True
            
            # Always keep final conv head trainable
            for p in self.backbone._conv_head.parameters():
                p.requires_grad = True
            for p in self.backbone._bn1.parameters():
                p.requires_grad = True

        # Temporal Module (Bidirectional GRU)
        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3  # Added dropout for regularization
        )
        self.temporal_dim = hidden_size * 2  # *2 for bidirectional

        # Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_dim * 2, 512),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_temporal=False):
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
            return_temporal: if True, also return temporal features for visualization
        
        Returns:
            logits: (B, num_classes)
            temporal_out: (B, T, temporal_dim) - only if return_temporal=True
        """
        B, T, C, H, W = x.shape
        
        # Process all frames through CNN
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)  # (B*T, feature_dim, 1, 1)
        features = features.view(B, T, self.feature_dim)  # (B, T, feature_dim)

        # Temporal modeling with GRU
        temporal_out, _ = self.temporal(features)  # (B, T, temporal_dim)

        # Aggregate temporal features (mean + max pooling)
        pooled = torch.cat([
            temporal_out.mean(dim=1),      # (B, temporal_dim)
            temporal_out.max(dim=1)[0]     # (B, temporal_dim)
        ], dim=1)  # (B, temporal_dim*2)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        if return_temporal:
            return logits, temporal_out
        return logits

    def get_feature_extractor_layer(self, layer_name='final'):
        """
        Get specific layer from feature extractor for Grad-CAM.
        
        Args:
            layer_name: 'final', 'block_6', 'block_5', etc.
        
        Returns:
            target_layer for Grad-CAM hooks
        """
        if layer_name == 'final':
            # Final conv layer (best for overall classification)
            return self.backbone._conv_head
        elif layer_name.startswith('block_'):
            # Get specific block (for multi-scale Grad-CAM)
            block_idx = int(layer_name.split('_')[1])
            return self.backbone._blocks[block_idx]
        else:
            # Default to final conv
            return self.backbone._conv_head


# =========================
# Model Information
# =========================
def print_model_info():
    """Print model architecture details"""
    model = TemporalCNN_GRU(num_classes=2, freeze_backbone=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*60)
    print("MODEL ARCHITECTURE: EfficientNet-B0 + BiGRU")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Feature dimension: 1280")
    print(f"Temporal dimension: 1024 (512*2 bidirectional)")
    print("="*60)
    
    # Test forward pass
    dummy_input = torch.randn(2, 16, 3, 224, 224)  # (B=2, T=16, C=3, H=224, W=224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("="*60)


if __name__ == "__main__":
    print_model_info()