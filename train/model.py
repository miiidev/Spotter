import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at 3 scales for better spatial resolution"""
    def __init__(self, backbone):
        super().__init__()
        
        # Low-level features (28×28) - fine texture details
        self.block3 = nn.Sequential(
            backbone._conv_stem,
            backbone._bn0,
            backbone._swish,
            *backbone._blocks[:7],  # Up to block 3
        )
        
        # Mid-level features (14×14) - facial structure
        self.block5 = nn.Sequential(
            *backbone._blocks[7:12],  # Blocks 4-5
        )
        
        # High-level features (7×7) - semantic understanding
        self.block7 = nn.Sequential(
            *backbone._blocks[12:],  # Blocks 6-7
            backbone._conv_head,
            backbone._bn1,
            backbone._swish,
        )
        
        # Fusion conv: will be initialized based on backbone type
        self.fusion = None  # Will be initialized based on backbone type
        
    def _init_fusion(self, in_channels, out_channels):
        """Initialize fusion layer based on backbone"""
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        # x: (B*T, 3, H, W) where H,W can be 224 or 300
        feat_low = self.block3(x)    # (B*T, 40 or 48, 28, 28) - actually depends on input size
        feat_mid = self.block5(feat_low)  # (B*T, 112 or 136, 14, 14)
        feat_high = self.block7(feat_mid)  # (B*T, 1280 or 1536, 7, 7)
        
        # Get the actual spatial dimensions (they may vary with input size)
        target_h, target_w = feat_low.shape[2], feat_low.shape[3]
        
        # Upsample mid and high to match feat_low spatial dimensions
        feat_mid_up = nn.functional.interpolate(
            feat_mid, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        feat_high_up = nn.functional.interpolate(
            feat_high, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        
        # Concatenate and fuse
        fused = torch.cat([feat_low, feat_mid_up, feat_high_up], dim=1)
        
        # Initialize fusion layer on first forward pass
        if self.fusion is None:
            in_channels = fused.shape[1]
            out_channels = feat_high.shape[1]  # Use high-level feature dim
            self._init_fusion(in_channels, out_channels)
            if x.is_cuda:
                self.fusion = self.fusion.cuda()
        
        output = self.fusion(fused)  # (B*T, 1280 or 1536, 1, 1)
        
        return output, feat_high  # Return both fused features and high-level for Grad-CAM


class SpatialAttentionModule(nn.Module):
    """Learn to focus on deepfake-prone facial regions"""
    def __init__(self, in_channels=1280, reduction=8):
        super().__init__()
        
        # Channel attention branch
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention branch
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B*T, C, H, W)
        
        # Channel attention
        c_attn = self.channel_attn(x)  # (B*T, C, 1, 1)
        x = x * c_attn  # Weight channels
        
        # Spatial attention  
        s_attn = self.spatial_attn(x)  # (B*T, 1, H, W)
        x_attended = x * s_attn  # Weight spatial locations
        
        return x_attended, s_attn  # Return both features and spatial attention map


class TemporalCNN_GRU(nn.Module):
    def __init__(self, num_classes=2, hidden_size=512, freeze_backbone=True, backbone='efficientnet-b0', use_multiscale=False):
        super().__init__()

        # Upgrade backbone - support B0 and B3
        self.backbone = EfficientNet.from_pretrained(backbone)
        self.backbone_name = backbone
        self.use_multiscale = use_multiscale
        
        # Feature dimension based on backbone
        backbone_dims = {
            'efficientnet-b0': 1280,
            'efficientnet-b3': 1536,
            'efficientnet-b4': 1792,
        }
        self.feature_dim = backbone_dims.get(backbone, 1280)
        
        # Choose between multi-scale or standard feature extraction
        if use_multiscale:
            self.feature_extractor = MultiScaleFeatureExtractor(self.backbone)
        else:
            # Standard feature extraction (backward compatible)
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

        # Add spatial attention after feature extraction
        self.spatial_attention = SpatialAttentionModule(in_channels=self.feature_dim)
        
        # Freeze/unfreeze strategy
        if freeze_backbone:
            # Freeze all backbone layers
            for p in self.backbone.parameters():
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
            
            # Spatial attention is always trainable
            for p in self.spatial_attention.parameters():
                p.requires_grad = True
            
            # If using multiscale, keep fusion layers trainable
            if use_multiscale:
                if hasattr(self.feature_extractor, 'fusion') and self.feature_extractor.fusion is not None:
                    for p in self.feature_extractor.fusion.parameters():
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

    def forward(self, x, return_temporal=False, return_attention=False):
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
            return_temporal: if True, also return temporal features for visualization
            return_attention: if True, also return spatial attention maps
        
        Returns:
            logits: (B, num_classes)
            temporal_out: (B, T, temporal_dim) - only if return_temporal=True
            spatial_attn_maps: (B, T, H, W) - only if return_attention=True
        """
        B, T, C, H, W = x.shape
        
        # Process all frames through CNN
        x = x.view(B * T, C, H, W)
        
        # Extract features (multiscale or standard)
        if self.use_multiscale:
            features, feat_high = self.feature_extractor(x)  # (B*T, feature_dim, 1, 1), (B*T, feature_dim, 7, 7)
            features_spatial = feat_high  # Use high-level features for spatial attention
        else:
            features = self.feature_extractor(x)  # (B*T, feature_dim, 1, 1)
            # For standard extraction, we need to get features before pooling
            # Re-extract features before pooling for spatial attention
            x_temp = x
            for layer in list(self.feature_extractor)[:-1]:  # All except AdaptiveAvgPool2d
                x_temp = layer(x_temp)
            features_spatial = x_temp  # (B*T, feature_dim, H, W)
        
        # Apply spatial attention
        features_attended, spatial_attn_maps = self.spatial_attention(features_spatial)
        
        # Pool attended features if not already pooled
        if features_attended.dim() == 4 and features_attended.shape[2] > 1:
            features_pooled = nn.AdaptiveAvgPool2d(1)(features_attended)
        else:
            features_pooled = features_attended
        
        # If using multiscale, combine with original features
        if self.use_multiscale:
            features_final = features_pooled  # Already fused
        else:
            features_final = features_pooled
        
        features_final = features_final.view(B, T, self.feature_dim)  # (B, T, feature_dim)

        # Temporal modeling with GRU
        temporal_out, _ = self.temporal(features_final)  # (B, T, temporal_dim)

        # Aggregate temporal features (mean + max pooling)
        pooled = torch.cat([
            temporal_out.mean(dim=1),      # (B, temporal_dim)
            temporal_out.max(dim=1)[0]     # (B, temporal_dim)
        ], dim=1)  # (B, temporal_dim*2)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        if return_attention:
            # Return spatial attention maps reshaped
            h, w = spatial_attn_maps.shape[2], spatial_attn_maps.shape[3]
            return logits, spatial_attn_maps.view(B, T, h, w)
        if return_temporal:
            return logits, temporal_out
        return logits

    def get_multiscale_layers(self):
        """
        Get layers for multi-scale Grad-CAM.
        
        Returns:
            dict of layer names to layers for Grad-CAM hooks
        """
        if self.use_multiscale:
            return {
                'fine': self.feature_extractor.block3[-1],    # 28×28
                'mid': self.feature_extractor.block5[-1],     # 14×14
                'semantic': self.feature_extractor.block7[0]  # 7×7
            }
        else:
            # For backward compatibility
            return {
                'semantic': self.backbone._conv_head
            }

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
def print_model_info(backbone='efficientnet-b0', use_multiscale=False):
    """Print model architecture details"""
    model = TemporalCNN_GRU(
        num_classes=2, 
        freeze_backbone=True, 
        backbone=backbone,
        use_multiscale=use_multiscale
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*60)
    print(f"MODEL ARCHITECTURE: {backbone.upper()} + BiGRU + Spatial Attention")
    if use_multiscale:
        print("Multi-Scale Feature Extraction: ENABLED")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Temporal dimension: 1024 (512*2 bidirectional)")
    print(f"Spatial Attention: ENABLED")
    print("="*60)
    
    # Test forward pass with appropriate input size
    input_size = 300 if 'b3' in backbone else 224
    dummy_input = torch.randn(2, 16, 3, input_size, input_size)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("="*60)


if __name__ == "__main__":
    print("\n=== Testing EfficientNet-B0 (Standard) ===")
    print_model_info(backbone='efficientnet-b0', use_multiscale=False)
    
    print("\n=== Testing EfficientNet-B3 (Multi-Scale) ===")
    print_model_info(backbone='efficientnet-b3', use_multiscale=True)