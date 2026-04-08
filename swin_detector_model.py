# swin_v2.py
# Swin Transformer with Mask2Former-style decoder and multi-token texture integration

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_, DropPath
import timm
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def _choose_gn_groups(channels, max_groups=32):
    """Find the largest number of groups that divides channels."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1  # Fallback


class SinePositionalEncoding2D(nn.Module):
    """
    2D sine-cosine positional encodings that work for arbitrary HxW.
    Output has 2*num_pos_feats channels (num_pos_feats for y, num_pos_feats for x).
    """
    def __init__(self, num_pos_feats: int):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = 10000

    def forward(self, x):  # x: (B,C,H,W)
        device = x.device
        B, C, H, W = x.shape
        
        # Create positional grids normalized to [0, 1]
        y_embed = torch.linspace(0, 1, steps=H, device=device).view(H, 1).expand(H, W)  # (H, W)
        x_embed = torch.linspace(0, 1, steps=W, device=device).view(1, W).expand(H, W)  # (H, W)
        
        # Scale to more reasonable range
        y_embed = y_embed * 2 * torch.pi
        x_embed = x_embed * 2 * torch.pi
        
        # Create frequency bands (only half of num_pos_feats, will be used for both sin and cos)
        num_feats = self.num_pos_feats // 2
        dim_t = torch.arange(num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * dim_t / self.num_pos_feats)
        
        # Compute positional encodings for y dimension
        pos_y = y_embed.unsqueeze(-1) / dim_t  # (H, W, num_feats)
        pos_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=3).flatten(2)  # (H, W, num_pos_feats)
        
        # Compute positional encodings for x dimension
        pos_x = x_embed.unsqueeze(-1) / dim_t  # (H, W, num_feats)
        pos_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=3).flatten(2)  # (H, W, num_pos_feats)
        
        # Concatenate y and x encodings: total 2*num_pos_feats channels
        pe = torch.cat([pos_y, pos_x], dim=2)  # (H, W, 2*num_pos_feats)
        
        # Reshape to (B, C, H, W) format
        pe = pe.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2*num_pos_feats, H, W)
        
        return pe


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute positional embeddings, learned. Size is now configurable.
    """
    def __init__(self, num_pos_feats=128, max_h=56, max_w=56):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, num_pos_feats)
        self.col_embed = nn.Embedding(max_w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        # Assert that the input size does not exceed the embedding table size
        assert h <= self.row_embed.num_embeddings, \
            f"Height of feature map ({h}) exceeds max_h ({self.row_embed.num_embeddings}) in PositionEmbeddingLearned."
        assert w <= self.col_embed.num_embeddings, \
            f"Width of feature map ({w}) exceeds max_w ({self.col_embed.num_embeddings}) in PositionEmbeddingLearned."

        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}x{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# NEW: Simplified backbone using official timm model
class TimmSwinBackbone(nn.Module):
    """Robust backbone using timm's official Swin implementation."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        model_name = config.get('pretrained_model_name', 'swin_base_patch4_window7_224')
        img_size = config.get('img_size', 224)
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=config.get('pretrained', True),
            features_only=True,
            img_size=img_size,
        )
        
        self.feature_channels = self.backbone.feature_info.channels()
        
        logger.info(f"Loaded timm Swin backbone: {model_name}")
        logger.info(f"Image size: {img_size}")
        logger.info(f"Feature channels: {self.feature_channels}")
        
    def forward(self, x):
        # timm features_only returns a list of feature tensors
        feats = self.backbone(x)
        # Detect and convert NHWC to NCHW if needed
        normed = []
        for i, f in enumerate(feats):
            if f.dim() == 4:
                expected_channels = self.feature_channels[i]
                
                # Check if last dimension matches expected channels (NHWC indicator)
                # and second dimension doesn't match (not NCHW)
                if f.shape[-1] == expected_channels and f.shape[1] != expected_channels:
                    # NHWC detected: (B, H, W, C), convert to NCHW
                    f = f.permute(0, 3, 1, 2).contiguous()
            normed.append(f)
        return normed

class MultiScalePixelDecoder(nn.Module):
    """
    Mask2Former-style pixel decoder:
    - Projects all backbone stages to a common dim
    - Top-down fusion (FPN-like)
    - Exposes multiple mask feature levels (highest K resolutions)
    - Returns: memory_feats for transformer, mask_features list for dot-product
    """
    def __init__(self, in_dims, embed_dim, num_mask_levels=2):
        super().__init__()
        self.num_mask_levels = num_mask_levels
        # 1x1 projection of all backbone features to embed_dim
        self.input_proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_dims])
        # Lateral refinement per level (3x3 + GN + ReLU)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                nn.GroupNorm(_choose_gn_groups(embed_dim), embed_dim),
                nn.ReLU(inplace=True),
            ) for _ in in_dims
        ])
        # Output mask feature heads for the highest resolutions
        self.mask_feature_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                nn.GroupNorm(_choose_gn_groups(embed_dim), embed_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_mask_levels)
        ])
        # Learned level embeddings for transformer memory
        self.level_embed = nn.Embedding(len(in_dims), embed_dim)

    def forward(self, feats):
        # feats: list of tensors [B,C_i,H_i,W_i] from high->low or low->high depending on timm order
        # Project to common dim
        proj = [self.input_proj[i](feats[i]) for i in range(len(feats))]
        # Build top-down features: ensure order is from lowest->highest for fusion
        out = [None] * len(proj)
        out[-1] = proj[-1]
        for i in range(len(proj) - 2, -1, -1):
            up = F.interpolate(out[i + 1], size=proj[i].shape[-2:], mode="bilinear", align_corners=False)
            out[i] = proj[i] + up
        # Lateral refinement for memory per level
        memory_feats = []
        for i, x in enumerate(out):
            memory_feats.append(self.lateral_convs[i](x))
        # Mask features from the highest K resolutions (index 0 is highest)
        high_to_low = memory_feats  # after fusion, index 0 is highest resolution
        mask_srcs = high_to_low[: self.num_mask_levels]
        mask_features = [self.mask_feature_convs[i](x) for i, x in enumerate(mask_srcs)]
        return memory_feats, mask_features, self.level_embed.weight


class SwinMask2Former(nn.Module):
    """Mask2Former-style model with multi-scale pixel decoder and multi-token texture integration."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 1. Swin Transformer Backbone
        self.backbone = TimmSwinBackbone(config)
        
        # 2. Multi-Scale Pixel Decoder
        transformer_dim = config.get('transformer_dim', 256)
        self.num_mask_feature_levels = config.get('num_mask_feature_levels', 2)
        
        swin_dims = self.backbone.feature_channels
        self.pixel_decoder = MultiScalePixelDecoder(
            in_dims=swin_dims, embed_dim=transformer_dim, num_mask_levels=self.num_mask_feature_levels
        )
        
        # Per-level sinusoidal positional encoders (no size assumptions)
        num_stages = len(swin_dims)  # 4 for Swin-T
        self.per_level_pos = nn.ModuleList()
        for _ in range(num_stages):
            self.per_level_pos.append(SinePositionalEncoding2D(num_pos_feats=transformer_dim // 2))
        
        # 3. Transformer Components
        num_queries = config.get('num_queries', 100)
        num_heads = config.get('transformer_nhead', 8)
        num_layers = config.get('transformer_layers', 6)

        self.query_embed = nn.Embedding(num_queries, transformer_dim)
        
        # FiLM texture modulation (local texture maps only)
        self.use_glcm = config.get('use_glcm', False)
        if self.use_glcm:
            # FiLM over local texture map projected to per-stage channels
            self.use_texture_film = True
            # Use LazyConv2d to infer texture channels at first forward
            self.texture_proj_s1 = nn.LazyConv2d(out_channels=transformer_dim, kernel_size=1)
            self.texture_proj_s2 = nn.LazyConv2d(out_channels=transformer_dim, kernel_size=1)
            self.film_s1 = nn.Conv2d(transformer_dim, 2 * transformer_dim, kernel_size=1)
            self.film_s2 = nn.Conv2d(transformer_dim, 2 * transformer_dim, kernel_size=1)
            # FiLM calibration: LayerNorm and learnable gates
            self.norm_s1 = nn.GroupNorm(_choose_gn_groups(transformer_dim), transformer_dim)
            self.norm_s2 = nn.GroupNorm(_choose_gn_groups(transformer_dim), transformer_dim)
            self.film_gate_s1 = nn.Parameter(torch.tensor(0.5))
            self.film_gate_s2 = nn.Parameter(torch.tensor(0.5))
        else:
            self.use_texture_film = False
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim, 
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. Prediction Heads
        num_classes = config.get('num_classes', 1)
        self.class_head = nn.Linear(transformer_dim, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim), nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim), nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim)
        )

    def forward(self, pixel_values=None, texture_maps=None, pixel_mask=None, _swin_features=None):
        # 1. Get multi-scale features from the backbone
        swin_features = _swin_features if _swin_features is not None else self.backbone(pixel_values)
        
        # 2. Multi-scale memory + K mask feature levels
        memory_feats, mask_features, level_embed = self.pixel_decoder(swin_features)
        
        # 3. Apply FiLM modulation with normalization and gating to early memory features
        if self.use_texture_film and texture_maps is not None and texture_maps.numel() > 0:
            B = texture_maps.shape[0]
            # Stage 0 (highest res) FiLM with calibration
            memory_feats[0] = self.norm_s1(memory_feats[0])
            t1 = F.interpolate(texture_maps, size=memory_feats[0].shape[-2:], mode="bilinear", align_corners=False)
            t1 = self.texture_proj_s1(t1)                              # (B, C, H, W)
            gamma_beta = self.film_s1(t1)                              # (B, 2C, H, W)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            gate1 = torch.sigmoid(self.film_gate_s1)
            memory_feats[0] = memory_feats[0] * (1 + gate1 * torch.tanh(gamma)) + gate1 * beta

            # Stage 1 FiLM (next highest res) with calibration
            if len(memory_feats) > 1:
                memory_feats[1] = self.norm_s2(memory_feats[1])
                t2 = F.interpolate(texture_maps, size=memory_feats[1].shape[-2:], mode="bilinear", align_corners=False)
                t2 = self.texture_proj_s2(t2)
                gamma_beta = self.film_s2(t2)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
                gate2 = torch.sigmoid(self.film_gate_s2)
                memory_feats[1] = memory_feats[1] * (1 + gate2 * torch.tanh(gamma)) + gate2 * beta
        
        # 4. Build transformer memory: concat all levels with per-level sine PE + level embed
        memories = []
        for lvl, feat in enumerate(memory_feats):
            pos = self.per_level_pos[lvl](feat)                          # (B, C, H, W) with C=transformer_dim
            mem = (feat + pos).flatten(2).permute(0, 2, 1)               # (B, HW, C)
            mem = mem + level_embed[lvl].view(1, 1, -1)                  # level encoding
            memories.append(mem)
        memory = torch.cat(memories, dim=1)                              # (B, sum(HW_l), C)
        
        # 5. Queries and decoder
        B = memory.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)    # (B,Q,C)
        decoder_output = self.decoder(tgt=query_embed, memory=memory)          # (B,Q,C)
        
        # 6. Prediction heads
        class_logits = self.class_head(decoder_output)                         # (B,Q,Cls+1)
        mask_embeddings = self.mask_head(decoder_output)                       # (B,Q,C)
        
        # 7. Multi-level mask logits: sum of dot-products at K highest resolutions
        pred_masks = None
        target_size = mask_features[0].shape[-2:]
        for i, mf in enumerate(mask_features):
            logits_i = torch.einsum("bqc,bchw->bqhw", mask_embeddings, mf)     # (B,Q,H_i,W_i)
            if mf.shape[-2:] != target_size:
                logits_i = F.interpolate(logits_i, size=target_size, mode="bilinear", align_corners=False)
            pred_masks = logits_i if pred_masks is None else (pred_masks + logits_i)
        
        return {"pred_logits": class_logits, "pred_masks": pred_masks}

    def get_config(self):
        """Return model configuration for logging and checkpointing"""
        return dict(self.config)

