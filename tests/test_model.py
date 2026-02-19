"""
Unit tests for model architecture improvements.

Tests for:
- MultiScaleFeatureExtractor
- SpatialAttentionModule
- TemporalCNN_GRU with B0 and B3
- Multi-scale Grad-CAM
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.model import TemporalCNN_GRU, MultiScaleFeatureExtractor, SpatialAttentionModule


class TestMultiScaleFeatureExtractor(unittest.TestCase):
    """Test multi-scale feature extraction"""
    
    def test_forward_pass_b0(self):
        """Test forward pass with EfficientNet-B0"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0', use_multiscale=True)
        
        # Test with B0 input size (224x224)
        batch_size, num_frames = 2, 16
        x = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 2), "Output shape incorrect for B0")
        
    def test_forward_pass_b3(self):
        """Test forward pass with EfficientNet-B3"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b3', use_multiscale=True)
        
        # Test with B3 input size (300x300)
        batch_size, num_frames = 2, 16
        x = torch.randn(batch_size, num_frames, 3, 300, 300)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 2), "Output shape incorrect for B3")
        self.assertEqual(model.feature_dim, 1536, "B3 should have 1536 features")
        
    def test_multiscale_layers(self):
        """Test that multi-scale layers are properly registered"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0', use_multiscale=True)
        
        layers = model.get_multiscale_layers()
        
        self.assertIn('fine', layers, "Missing fine-scale layer")
        self.assertIn('mid', layers, "Missing mid-scale layer")
        self.assertIn('semantic', layers, "Missing semantic-scale layer")


class TestSpatialAttentionModule(unittest.TestCase):
    """Test spatial attention module"""
    
    def test_attention_output_shape(self):
        """Test that attention module outputs correct shapes"""
        # Test with B0 dimensions
        spatial_attn = SpatialAttentionModule(in_channels=1280)
        
        # Input: (B*T, C, H, W)
        x = torch.randn(32, 1280, 7, 7)  # 32 frames, 1280 channels, 7x7 spatial
        
        with torch.no_grad():
            x_attended, attn_maps = spatial_attn(x)
        
        self.assertEqual(x_attended.shape, (32, 1280, 7, 7), "Attended features shape incorrect")
        self.assertEqual(attn_maps.shape, (32, 1, 7, 7), "Attention maps shape incorrect")
        
    def test_attention_values(self):
        """Test that attention values are in [0, 1] range"""
        spatial_attn = SpatialAttentionModule(in_channels=1280)
        
        x = torch.randn(8, 1280, 7, 7)
        
        with torch.no_grad():
            _, attn_maps = spatial_attn(x)
        
        self.assertTrue(torch.all(attn_maps >= 0), "Attention values should be >= 0")
        self.assertTrue(torch.all(attn_maps <= 1), "Attention values should be <= 1")


class TestTemporalCNNGRU(unittest.TestCase):
    """Test TemporalCNN_GRU model"""
    
    def test_backward_compatibility_b0(self):
        """Test that B0 model works without multi-scale (backward compatible)"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0', use_multiscale=False)
        
        batch_size, num_frames = 2, 16
        x = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 2))
        self.assertEqual(model.feature_dim, 1280)
        
    def test_return_temporal_features(self):
        """Test returning temporal features"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0')
        
        batch_size, num_frames = 2, 16
        x = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        with torch.no_grad():
            logits, temporal_out = model(x, return_temporal=True)
        
        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertEqual(temporal_out.shape, (batch_size, num_frames, 1024))  # 512*2 bidirectional
        
    def test_return_attention_maps(self):
        """Test returning spatial attention maps"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0')
        
        batch_size, num_frames = 2, 16
        x = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        with torch.no_grad():
            logits, attn_maps = model(x, return_attention=True)
        
        self.assertEqual(logits.shape, (batch_size, 2))
        # Attention maps should have shape (B, T, H, W)
        self.assertEqual(len(attn_maps.shape), 4)
        self.assertEqual(attn_maps.shape[0], batch_size)
        self.assertEqual(attn_maps.shape[1], num_frames)
        
    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0', freeze_backbone=False)
        
        batch_size, num_frames = 2, 8
        x = torch.randn(batch_size, num_frames, 3, 224, 224, requires_grad=True)
        labels = torch.tensor([0, 1])
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad, "Gradients should flow to input")
        
        # Check that some model parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad, "Some model parameters should have gradients")


class TestModelDimensions(unittest.TestCase):
    """Test model dimensions for different configurations"""
    
    def test_b0_dimensions(self):
        """Test B0 model dimensions"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0')
        self.assertEqual(model.feature_dim, 1280)
        
    def test_b3_dimensions(self):
        """Test B3 model dimensions"""
        model = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b3')
        self.assertEqual(model.feature_dim, 1536)
        
    def test_total_parameters(self):
        """Test that parameter count is reasonable"""
        model_b0 = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b0')
        model_b3 = TemporalCNN_GRU(num_classes=2, backbone='efficientnet-b3')
        
        params_b0 = sum(p.numel() for p in model_b0.parameters())
        params_b3 = sum(p.numel() for p in model_b3.parameters())
        
        # B0 should have around 17M params, B3 around 25M
        self.assertGreater(params_b0, 10_000_000, "B0 should have > 10M params")
        self.assertLess(params_b0, 30_000_000, "B0 should have < 30M params")
        
        self.assertGreater(params_b3, 20_000_000, "B3 should have > 20M params")
        self.assertLess(params_b3, 40_000_000, "B3 should have < 40M params")
        
        self.assertGreater(params_b3, params_b0, "B3 should have more params than B0")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
