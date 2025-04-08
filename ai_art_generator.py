import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Set the title
st.title("🖼️ AI Virtual Art Generator")

# Text input
prompt = st.text_input("Enter your prompt:")

# Load model only once
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")  # or use 'cuda' if you have GPU
    return pipe

pipe = load_pipeline()

# Generate image
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Artwork")
            image.save("generated_art.png")
    else:
        st.warning("Please enter a prompt first.")
