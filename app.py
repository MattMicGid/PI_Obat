import os
import gdown
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from gtts import gTTS
import tensorflow as tf
import tempfile
import base64
from io import BytesIO

# ========== KONFIGURASI ========== #
st.set_page_config(page_title="💊 ObatVision", layout="centered")
st.title("💊 ObatVision - Deteksi Obat Lewat Gambar")

GOOGLE_DRIVE_CONFIG = {
    "model_file_id": "1WEALsJVVZjTedzapj0ykmzVg3wf4-Yub",
    "dataset_file_id": "1V-HI64YbBUQmkd20IOqMzAEk88PlqECw",
    "model_filename": "model_obat_resnet152v2_100.h5",
    "dataset_filename": "dataset_obat.csv"
}

# ========== DOWNLOAD FILE JIKA BELUM ADA ========== #
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

download_from_drive(GOOGLE_DRIVE_CONFIG["model_file_id"], GOOGLE_DRIVE_CONFIG["model_filename"])
download_from_drive(GOOGLE_DRIVE_CONFIG["dataset_file_id"], GOOGLE_DRIVE_CONFIG["dataset_filename"])

# ========== LOAD MODEL DAN DATASET ========== #
model = tf.keras.models.load_model(GOOGLE_DRIVE_CONFIG["model_filename"])
dataset = pd.read_csv(GOOGLE_DRIVE_CONFIG["dataset_filename"])

# ========== FUNGSI PENDUKUNG ========== #
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0
    return img_array

def play_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_bytes = open(fp.name, 'rb').read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def show_info_popup(title, text):
    with st.expander(title):
        st.markdown(text)
        play_audio(text)

# ========== MENU UTAMA ========== #
st.subheader("Silakan unggah atau ambil gambar obat")
img_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
camera_img = st.camera_input("Atau Ambil Gambar Lewat Kamera")

input_image = None
if img_file:
    input_image = Image.open(img_file)
elif camera_img:
    input_image = Image.open(camera_img)

if input_image:
    st.image(input_image, caption="Gambar Obat", use_column_width=True)

    with st.spinner("🔍 Mendeteksi obat..."):
        img_array = preprocess_image(input_image)
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        obat_row = dataset.iloc[predicted_index]
        nama_obat = obat_row["nama_obat"]

        st.success(f"✅ Prediksi: **{nama_obat}** ({confidence:.2f}% confidence)")

        # === INFO UTAMA ===
        st.markdown(f"""
        **Golongan:** {obat_row['golongan']}  
        **Jenis:** {obat_row['jenis']}  
        **Manfaat:** {obat_row['manfaat']}  
        **Aturan Minum:** {obat_row['aturan_minum']}  
        **Catatan:** {obat_row['catatan']}
        """)
        
        peringatan = "Aturan minum dapat berbeda-beda pada setiap orang, harus mengikuti saran dari dokter yang sudah cek kondisi pasien."
        st.warning(peringatan)
        play_audio(peringatan)

        st.markdown("### Lihat lebih lanjut:")
        show_info_popup("📌 Efek Samping", obat_row["efek_samping"])
        show_info_popup("🥦 Pantangan Makanan", obat_row["pantangan_makanan"])
        show_info_popup("❗ Interaksi Negatif", obat_row["interaksi_negatif"])
        show_info_popup("⏰ Jika Lupa Minum", obat_row["jika_lupa_minum"])
        show_info_popup("💾 Cara Penyimpanan", obat_row["penyimpanan"])
