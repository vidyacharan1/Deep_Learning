import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class transformer_block(nn.Module):
    def __init__(self, dim, num_heads):
        super(transformer_block, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, slots):
        # Apply self-attention
        x = x.permute(1, 0, 2)  # Reshape for Multihead Attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        # Cross attention with slots
        slots = slots.unsqueeze(0)  # Add batch dimension for broadcasting
        attn_output, _ = self.attn(slots, x, x)
        x = x + attn_output
        x = self.norm(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if downsample:
            self.downsampling = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsampling(x)
        out += residual
        out = self.relu(out)
        return out

class DiffusionDecoder(nn.Module):
    def __init__(self):
        super(DiffusionDecoder, self).__init__()
        self.diffuse = nn.Sequential(
            ResidualBlock(3, 64),
            ResidualBlock(64, 128, downsample=True),
            transformer_block(128, num_heads=8),
            ResidualBlock(128, 128),
            transformer_block(128, num_heads=8),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x, slots):
        return self.diffuse(x)

class SlotAttentionNetwork(nn.Module):
    def __init__(self, num_slots, num_iters, resolution):
        super(SlotAttentionNetwork, self).__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.resolution = resolution

        # Define your encoder and slot attention modules here (similar to your original implementation)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(64)
        self.dense_encode = nn.Linear(4, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.slot_attention = nn.MultiheadAttention(64, num_heads=1)
        self.decoder = DiffusionDecoder()

    def forward(self, input):
        x = self.cnn_encoder(input)
        x = x.permute(0, 2, 3, 1)
        temp = grid_embed(self.resolution).to(device)
        x = x + self.dense_encode(temp)
        x = x.view(-1, x.size(1) * x.size(2), x.size(-1))
        x = self.mlp(self.norm(x))
        
        # Apply slot attention
        # slots = x.unsqueeze(0)  # Add batch dimension for attention
        slots = x
        x, _ = self.slot_attention(slots, x, x)
        # shape is (seq_len, batch, embed_dim) but we got [1, 16384, 64]
        #  The size of tensor a (64) must match the size of tensor b (3) at non-singleton dimension 1
        x = x.view(-1, slots.shape[1], slots.shape[2])
        print(x.shape)
        # Decode using Diffusion Decoder
        print(input.shape)
        
        
        print(input.shape)
        reconstructed = self.decoder(input, x)
        
        return reconstructed

def grid_embed(resolution):
    scope = [np.linspace(0.0, 1.0, num=x) for x in resolution]
    grid = np.meshgrid(*scope, indexing='ij', sparse=False)
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, (resolution[0], resolution[1], -1))
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    # print(grid.shape)
    return torch.cat([torch.tensor(grid), 1.0 - torch.tensor(grid)], dim=-1)


# sample usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SlotAttentionNetwork(num_slots=11, num_iters=3, resolution=(128, 128)).to(device)
num_slots = 11
num_iters = 3
resolution = (128, 128)
input = torch.randn(1, 3, 128, 128).to(device)
output = model(input)
print(output.shape)