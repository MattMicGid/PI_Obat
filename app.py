import streamlit as st
from PIL import Image

st.set_page_config(page_title="Contoh App Sederhana", layout="centered")

st.title("🎈 Contoh Streamlit App Sederhana")

st.write("Masukkan nama kamu di bawah ini:")

name = st.text_input("Nama kamu:")

if name:
    st.success(f"Halo, {name}! Selamat datang di Streamlit 🎉")

uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
