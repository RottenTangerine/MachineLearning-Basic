import numpy as np
import torch
import torch.nn as nn
import os
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import Generator

os.makedirs("images", exist_ok=True)
img_shape = (1, 28, 28)
# hyper params
epoch_num = 50
batch_size = 64
learning_rate = 2e-4
latent_dim = 2 ** 8


train_dataset = datasets.MNIST(root='../../data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


discriminator = Discriminator().to(device)
generator = Generator().to(device)
criterion = nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


for epoch in range(epoch_num):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)

        valid = Tensor(imgs.size(0), 1).fill_(1.0).detach()  # detach means "requires_grad = False"
        fake = Tensor(imgs.size(0), 1).fill_(0.0).detach()

        # Train Generator
        optimizer_G.zero_grad()

        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
        gen_imgs = generator(z)

        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch + 1}/ {epoch_num}] [Batch {i + 1}/ {len(train_loader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        batches_done = epoch * len(train_loader) + i
        if batches_done % 200 == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

torch.save(generator.state_dict(), 'generator_params.ckpt')
torch.save(discriminator.state_dict(), 'discriminator_params.ckpt')