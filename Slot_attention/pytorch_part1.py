import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots, num_iters, slot_size, mlp_hidden_dim, eps=1e-8):
        super(SlotAttentionModule, self).__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.slot_size = slot_size
        self.eps = eps
        self.mlp_hidden_dim = mlp_hidden_dim

        self.input_norm = nn.LayerNorm(slot_size)
        self.mu_slot = nn.Parameter(torch.randn(1, 1, slot_size))
        self.sigma_slot = nn.Parameter(torch.randn(1, 1, slot_size))
        self.k_proj = nn.Linear(slot_size, slot_size, bias=False)
        self.q_proj = nn.Linear(slot_size, slot_size, bias=False)
        self.v_proj = nn.Linear(slot_size, slot_size, bias=False)
        self.slots_norm = nn.LayerNorm(slot_size)
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_size)
        )
        self.mlp_norm = nn.LayerNorm(slot_size)

    def forward(self, x):
        inputs = self.input_norm(x)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        slots = self.mu_slot + torch.exp(self.sigma_slot) * torch.randn((x.size(0), self.num_slots, self.slot_size), device=x.device)

        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.slots_norm(slots)
            q = self.q_proj(slots)
            q *= torch.sqrt(torch.tensor(1/self.slot_size, dtype=torch.float32, device=x.device))
            attn = F.softmax(torch.bmm(k, q.transpose(1, 2)), dim=-1)
            attn = (attn + self.eps) / torch.sum(attn, dim=-2, keepdim=True)
            # print(attn.shape, v.shape)
            attn_sh = attn.permute(0, 2, 1)
            updates = torch.bmm(attn_sh, v)
            # print(updates.shape, slots_prev.shape)
            upd = updates.reshape(-1,self.slot_size)
            sl_pr = slots_prev.reshape(-1,self.slot_size)
            slots = self.gru(upd, sl_pr)
            # print(slots.shape, slots_prev.shape)
            slots_new = slots.reshape(-1, self.num_slots, self.slot_size)
            slots = slots_new + self.mlp(self.mlp_norm(slots_new))
        return slots
    
# samodule = SlotAttentionModule(11, 3, 64, 128)
# x = torch.randn(64,128*128,64)
# y = samodule(x)
# print(y.shape)


def grid_embed(resolution):
    scope = [np.linspace(0.0, 1.0, num=x) for x in resolution]
    grid = np.meshgrid(*scope, indexing='ij', sparse=False)
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, (resolution[0], resolution[1], -1))
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    # print(grid.shape)
    return torch.cat([torch.tensor(grid), 1.0 - torch.tensor(grid)], dim=-1)

# print(grid_embed((128, 128)).shape)

class SlotAttentionNetwork(nn.Module):
    def __init__(self, num_slots, num_iters, resolution):
        super(SlotAttentionNetwork, self).__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.resolution = resolution

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
        self.slot_attention = SlotAttentionModule(num_slots, num_iters, 64, 128)
        self.decoder_size = (8, 8)
        self.dense_decode = nn.Linear(4, 64)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, 3, stride=1, padding=1)
        )

    def forward(self, input):
        x = self.cnn_encoder(input)
        x = x.permute(0, 2, 3, 1)
        temp = grid_embed(self.resolution)
        # temp = temp.view(1, -1, temp.size(-1))
        x += self.dense_encode(temp)
        x = x.view(-1, x.size(1) * x.size(2), x.size(-1))
        x = self.mlp(self.norm(x))
        slots = self.slot_attention(x)
        x = slots.view(-1, slots.shape[2])[:, None, None, :]
        x = x.repeat(1, self.decoder_size[0], self.decoder_size[1], 1)
        x += self.dense_decode(grid_embed(self.decoder_size))
        x = x.permute(0, 3, 1, 2)
        x = self.cnn_decoder(x)
        recons, masks = torch.split(x, (3, 1), dim=1)
        masks = torch.sigmoid(masks)
        combined = torch.sum(recons * masks, dim=1)
        return recons, masks, combined, slots
    
# model = SlotAttentionNetwork(11, 3, (128, 128))
# x = torch.randn(32, 3, 128, 128)
# recons, masks, combined, slots = model(x)
# print(recons.shape, masks.shape, combined.shape, slots.shape)



# hyperparameters
batch_size = 64
num_slots = 11
num_iters = 3
resolution = (128, 128)

model = SlotAttentionNetwork(num_slots, num_iters, resolution)


# The dataset is in the folder '/kaggle/input/clevrtex-images/train'

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
criterion = nn.MSELoss()

# The dataset is in the folder '/kaggle/input/clevrtex-images/train' and it contains 40000 images
# The images should be loaded using the ImageFolder class and should be resized to 128*128*3
# The images are of size 240*320*3 and should be cropped to 128*128*3

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Define transform
transform = transforms.Compose([transforms.CenterCrop((128, 128)), transforms.ToTensor()])

# Load dataset
data_dir = '/kaggle/input/clevrtex-images/train/train'  
dataset = CustomDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Model training loop
for epoch in range(num_epochs):
    model.train()
    for images in dataloader:
        images = images.to(device)
        print(images.shape)
        recons, masks, combined, slots = model(images)
        loss = criterion(combined, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
    
    if (epoch % 3) == 0:
        torch.save(model.state_dict(), f'model_checkpoint_98_{epoch}.pt')

torch.save(model.state_dict(), 'model_A2_98.pt')





