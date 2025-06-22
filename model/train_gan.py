import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from gan import Generator

# Discriminador simple para GAN
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.img_shape = img_shape

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        validity = self.model(x)
        return validity

# Hiperparámetros
z_dim = 100
batch_size = 128
epochs = 2  # Puedes aumentar para mejor calidad

# Dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instanciar modelos
generator = Generator(z_dim=z_dim)
discriminator = Discriminator()
adversarial_loss = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Entrenamiento
for epoch in range(epochs):
    for imgs, labels in dataloader:
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)
        imgs, labels = imgs.to(device), labels.to(device)

        # Entrena Discriminador
        z = torch.randn(batch_size, z_dim, device=device)
        gen_labels = torch.randint(0, 10, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels)
        real_loss = adversarial_loss(discriminator(imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Entrena Generador
        z = torch.randn(batch_size, z_dim, device=device)
        gen_labels = torch.randint(0, 10, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Guarda el modelo entrenado
torch.save(generator.state_dict(), "generator.pth")
print("Modelo guardado como generator.pth en la carpeta actual.")
