import torch
from model.gan import Generator
import numpy as np
from PIL import Image

def generate_digit_images(digit, n_images=5):
    # Cargar modelo entrenado
    model = Generator()
    model.load_state_dict(torch.load("model/generator.pth", map_location="cpu"))
    model.eval()
    images = []
    for _ in range(n_images):
        # Generar ruido y etiqueta
        z = torch.randn(1, 100)
        label = torch.tensor([digit])
        with torch.no_grad():
            img = model(z, label)
        img = img.squeeze().numpy() * 127.5 + 127.5
        img = Image.fromarray(img.astype(np.uint8), mode="L")
        images.append(img)
    return images
