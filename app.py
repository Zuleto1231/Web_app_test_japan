# Web app principal (Streamlit)
import streamlit as st
from utils.generate import generate_digit_images

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
if st.button("Generate Images"):
    images = generate_digit_images(digit, n_images=5)
    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i, img in enumerate(images):
        cols[i].image(img, caption=f"Sample {i+1}", width=100)
