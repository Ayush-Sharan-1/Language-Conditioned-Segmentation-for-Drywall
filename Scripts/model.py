"""
Attention U-Net with FiLM conditioning for language-conditioned segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import clip


class FiLMLayer(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) layer for conditioning features with text embeddings.
    
    Applies: feature = gamma * feature + beta
    """
    
    def __init__(self, text_dim: int, feature_dim: int):
        """
        Args:
            text_dim: Dimension of text embedding
            feature_dim: Number of channels in feature map
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Linear layers to map text embedding to gamma and beta
        self.gamma_proj = nn.Linear(text_dim, feature_dim)
        self.beta_proj = nn.Linear(text_dim, feature_dim)
    
    def forward(self, features: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature map
            text_embedding: [B, text_dim] text embedding
        
        Returns:
            Conditioned features: [B, C, H, W]
        """
        # Project text embedding to gamma and beta
        gamma = self.gamma_proj(text_embedding)  # [B, C]
        beta = self.beta_proj(text_embedding)   # [B, C]
        
        # Reshape for broadcasting: [B, C, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM: feature = gamma * feature + beta
        conditioned = gamma * features + beta
        
        return conditioned


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections in U-Net.
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of channels in gating signal (decoder features)
            F_l: Number of channels in skip connection (encoder features)
            F_int: Intermediate number of channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder [B, F_g, H_g, W_g]
            x: Skip connection from encoder [B, F_l, H_l, W_l]
        
        Returns:
            Attention-weighted encoder features [B, F_l, H_l, W_l]
        """
        # Upsample gating signal to match spatial dimensions of x
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 spatial size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        # Add and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Compute attention weights
        psi = self.psi(psi)
        
        # Apply attention to skip connection
        return x * psi


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class CLIPTextEncoder(nn.Module):
    """
    Wrapper around CLIP text encoder.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device_str = device
        self.clip_model, _ = clip.load(model_name, device=device)
        
        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get text embedding dimension
        self.text_dim = self.clip_model.text_projection.shape[1]
    
    def forward(self, text: str) -> torch.Tensor:
        """
        Encode text prompt to embedding.
        
        Args:
            text: Text prompt string
        
        Returns:
            Text embedding [1, text_dim]
        """
        # Tokenize text
        text_tokens = clip.tokenize([text]).to(self.device_str)
        
        # Encode
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_tokens)
        
        return text_embedding.float()


class AttentionUNet(nn.Module):
    """
    Attention U-Net with FiLM conditioning for language-conditioned segmentation.
    """
    
    def __init__(self, in_channels: int = 3, text_dim: int = 512):
        """
        Args:
            in_channels: Number of input image channels (3 for RGB)
            text_dim: Dimension of text embedding from CLIP
        """
        super().__init__()
        
        self.text_dim = text_dim
        
        # Encoder (downsampling path)
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)
        
        # Decoder (upsampling path)
        # Input channels account for concatenation: decoder_feat + attention_gated_encoder_feat
        self.dec4 = ResidualBlock(1024 + 512, 512)  # bottleneck(1024) + att4(512)
        self.dec3 = ResidualBlock(512 + 256, 256)   # dec4(512) + att3(256)
        self.dec2 = ResidualBlock(256 + 128, 128)    # dec3(256) + att2(128)
        self.dec1 = ResidualBlock(128 + 64, 64)      # dec2(128) + att1(64)
        
        # Attention gates
        self.att4 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=128, F_l=64, F_int=32)
        
        # FiLM layers for each decoder stage
        self.film4 = FiLMLayer(text_dim, 512)
        self.film3 = FiLMLayer(text_dim, 256)
        self.film2 = FiLMLayer(text_dim, 128)
        self.film1 = FiLMLayer(text_dim, 64)
        
        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            text_embedding: Text embedding [B, text_dim]
        
        Returns:
            Logits [B, 1, H, W]
        """
        # Encoder path
        e1 = self.enc1(x)          # [B, 64, H, W]
        e1_pooled = self.pool(e1)  # [B, 64, H/2, W/2]
        
        e2 = self.enc2(e1_pooled)  # [B, 128, H/2, W/2]
        e2_pooled = self.pool(e2)  # [B, 128, H/4, W/4]
        
        e3 = self.enc3(e2_pooled)  # [B, 256, H/4, W/4]
        e3_pooled = self.pool(e3)  # [B, 256, H/8, W/8]
        
        e4 = self.enc4(e3_pooled)  # [B, 512, H/8, W/8]
        e4_pooled = self.pool(e4)  # [B, 512, H/16, W/16]
        
        # Bottleneck
        bottleneck = self.bottleneck(e4_pooled)  # [B, 1024, H/16, W/16]
        
        # Decoder path with attention gates and FiLM conditioning
        # Stage 4
        d4 = F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=False)
        att4 = self.att4(d4, e4)  # [B, 512, H/8, W/8]
        d4 = torch.cat([d4, att4], dim=1)  # [B, 1024+512=1536, H/8, W/8]
        d4 = self.dec4(d4)  # [B, 512, H/8, W/8]
        d4 = self.film4(d4, text_embedding)  # Apply FiLM conditioning
        
        # Stage 3
        d3 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        att3 = self.att3(d3, e3)  # [B, 256, H/4, W/4]
        d3 = torch.cat([d3, att3], dim=1)  # [B, 512+256=768, H/4, W/4]
        d3 = self.dec3(d3)  # [B, 256, H/4, W/4]
        d3 = self.film3(d3, text_embedding)  # Apply FiLM conditioning
        
        # Stage 2
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        att2 = self.att2(d2, e2)  # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, att2], dim=1)  # [B, 256+128=384, H/2, W/2]
        d2 = self.dec2(d2)  # [B, 128, H/2, W/2]
        d2 = self.film2(d2, text_embedding)  # Apply FiLM conditioning
        
        # Stage 1
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        att1 = self.att1(d1, e1)  # [B, 64, H, W]
        d1 = torch.cat([d1, att1], dim=1)  # [B, 128+64=192, H, W]
        d1 = self.dec1(d1)  # [B, 64, H, W]
        d1 = self.film1(d1, text_embedding)  # Apply FiLM conditioning
        
        # Final output
        output = self.final_conv(d1)  # [B, 1, H, W]
        
        return output


class LanguageConditionedSegmentationModel(nn.Module):
    """
    Complete model combining CLIP text encoder and Attention U-Net.
    """
    
    def __init__(self, clip_model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        
        self.device_str = device
        
        # Text encoder (frozen)
        self.text_encoder = CLIPTextEncoder(model_name=clip_model_name, device=device)
        text_dim = self.text_encoder.text_dim
        
        # Segmentation model
        self.segmentation_model = AttentionUNet(in_channels=3, text_dim=text_dim)
    
    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: Input image [B, 3, H, W]
            prompt: Text prompt string
        
        Returns:
            Logits [B, 1, H, W]
        """
        # Encode text prompt
        text_embedding = self.text_encoder(prompt)  # [1, text_dim]
        
        # Expand text embedding to batch size and move to same device as image
        batch_size = image.shape[0]
        text_embedding = text_embedding.expand(batch_size, -1).to(image.device)  # [B, text_dim]
        
        # Forward through segmentation model
        logits = self.segmentation_model(image, text_embedding)
        
        return logits
    
    def forward_with_embedding(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed text embedding.
        
        Args:
            image: Input image [B, 3, H, W]
            text_embedding: Pre-computed text embedding [1, text_dim] or [B, text_dim]
        
        Returns:
            Logits [B, 1, H, W]
        """
        # Expand text embedding to batch size if needed
        batch_size = image.shape[0]
        if text_embedding.shape[0] == 1:
            text_embedding = text_embedding.expand(batch_size, -1)
        text_embedding = text_embedding.to(image.device)  # [B, text_dim]
        
        # Forward through segmentation model
        logits = self.segmentation_model(image, text_embedding)
        
        return logits

