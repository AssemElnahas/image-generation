import streamlit as st
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(page_title="MNIST GAN Generator", page_icon="🎨", layout="centered")

# Define the Generator class exactly as it was in the training notebook
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh() # Final output mapped to range [-1, 1]
        )
        
    def forward(self, z):
        output = self.model(z)
        output = output.view(output.size(0), 1, 28, 28)
        return output

# Load the trained model using st.cache_resource so it only loads once
@st.cache_resource
def load_model():
    latent_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(latent_dim=latent_size).to(device)
    
    try:
        # Load weights safely, explicitly mapping to the current device (handles CPU vs GPU differences)
        model.load_state_dict(torch.load('generator.pth', map_location=device, weights_only=True))
        model.eval() # Set to evaluation mode
        return model, device, latent_size
    except FileNotFoundError:
        return None, device, latent_size

# UI Header
st.title("🎨 MNIST Handwritten Digit Generator")
st.write("This app uses a PyTorch Generative Adversarial Network (GAN) to create entirely new, fake handwritten digits from random noise. The model was trained from scratch on the MNIST dataset.")

# Attempt to load the model
model, device, latent_size = load_model()

if model is None:
    st.error("⚠️ **Model file (`generator.pth`) not found!**\n\nPlease make sure you run the training notebook completely and that it saves the model to this directory before using the app.")
else:
    # Sidebar controls
    st.sidebar.header("Generation Controls")
    num_images = st.sidebar.slider("Number of digits to generate", min_value=1, max_value=64, value=16, step=1)
    
    st.sidebar.write("---")

    # Generate Button
    if st.button("🚀 Generate New Digits", use_container_width=True):
        with st.spinner("The AI is generating digits from noise..."):
            with torch.no_grad():
                # 1. Generate random noise
                noise = torch.randn(num_images, latent_size).to(device)
                
                # 2. Pass noise through the generator
                generated_images = model(noise).cpu()
                
                # 3. Create a clean grid
                nrow = max(1, int(np.ceil(np.sqrt(num_images))))
                grid = torchvision.utils.make_grid(generated_images, nrow=nrow, normalize=True)
                
                # 4. Convert to numpy for matplotlib
                grid_np = np.transpose(grid.numpy(), (1, 2, 0))
                
                # 5. Plot and Display
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(grid_np)
                ax.axis("off")
                st.pyplot(fig)
