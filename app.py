import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from streamlit_extras.stylable_container import stylable_container
from gtts import gTTS
import tempfile
import requests

# ========== CONFIGURASI ==========
st.set_page_config(page_title="ObatVision", layout="centered")
st.title("💊 ObatVision: Deteksi dan Info Obat")

# Google Drive Config
GOOGLE_DRIVE_CONFIG = {
    "model_file_id": "1WEALsJVVZjTedzapj0ykmzVg3wf4-Yub",
    "dataset_file_id": "1V-HI64YbBUQmkd20IOqMzAEk88PlqECw",
    "model_filename": "model_obat_resnet152v2_100.h5",
    "dataset_filename": "dataset_obat.csv"
}

def download_from_gdrive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource(show_spinner=True)
def load_model():
    model_path = GOOGLE_DRIVE_CONFIG['model_filename']
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GOOGLE_DRIVE_CONFIG['model_file_id'], model_path)
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data(show_spinner=True)
def load_data():
    dataset_path = GOOGLE_DRIVE_CONFIG['dataset_filename']
    if not os.path.exists(dataset_path):
        st.info("Downloading dataset from Google Drive...")
        download_from_gdrive(GOOGLE_DRIVE_CONFIG['dataset_file_id'], dataset_path)
    df = pd.read_csv(dataset_path)
    return df

model = load_model()
obat_info_df = load_data()
class_names = sorted(obat_info_df['label'].unique())

# TTS menggunakan gtts
def speak(text):
    if not text.strip():
        return
    tts = gTTS(text=text, lang='id')
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        temp_audio = fp.name + ".mp3"
        tts.save(temp_audio)
        audio_bytes = open(temp_audio, 'rb').read()
        st.audio(audio_bytes, format='audio/mp3')

# ========== INPUT GAMBAR ==========
with st.sidebar:
    st.header("📷 Input Gambar")
    img_file = st.file_uploader("Unggah Gambar Obat", type=["jpg", "jpeg", "png"])
    camera_img = st.camera_input("Atau Ambil Gambar Realtime")
    img_input = img_file if img_file else camera_img

# ========== PROSES & PREDIKSI ==========
if img_input:
    img = Image.open(img_input).convert('RGB')
    img_resized = img.resize((256, 256))

    # Preprocessing
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Ambil info dari CSV
    info = obat_info_df[obat_info_df['label'] == predicted_label].iloc[0]

    # Output
    st.image(img, caption=f"Prediksi: {predicted_label} ({confidence:.2f}%)", use_column_width=True)

    st.subheader(f"💊 {info['nama_obat']}")
    st.markdown(f"""
    *Golongan:* {info['golongan']}  
    *Jenis:* {info['jenis']}  
    
    *Manfaat:* {info['manfaat']}  
    *Aturan Minum:* {info['aturan_minum']}  
    *Catatan:* {info['catatan']}  
    
    *🎯 Akurasi Prediksi:* {confidence:.2f}%
    """)

    st.warning("⚠ Aturan minum dapat berbeda-beda pada setiap orang. Ikuti saran dokter yang memahami kondisi Anda.")
    
    speak(f"Obat yang terdeteksi adalah {info['nama_obat']}. Aturan minum: {info['aturan_minum']}. Perhatian: {info['catatan']}")

    # ========== MENU LANJUTAN ==========
    with stylable_container("popup-menu", css="""
        button {
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            margin: 0.2rem 0;
        }
    """):
        st.markdown("📂 Lihat lebih lanjut:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Efek Samping"):
                st.info(f"Efek samping dari {info['nama_obat']}: {info.get('efek_samping', 'Tidak tersedia')}")
                speak(info.get('efek_samping', ''))
            if st.button("Pantangan Makanan"):
                st.info(info.get('pantangan_makanan', 'Tidak tersedia'))
                speak(info.get('pantangan_makanan', ''))
            if st.button("Interaksi Negatif"):
                st.info(info.get('interaksi_negatif', 'Tidak tersedia'))
                speak(info.get('interaksi_negatif', ''))
        with col2:
            if st.button("Jika Lupa Minum?"):
                st.info(info.get('jika_lupa_minum', 'Tidak tersedia'))
                speak(info.get('jika_lupa_minum', ''))
            if st.button("Cara Penyimpanan"):
                st.info(info.get('penyimpanan', 'Tidak tersedia'))
                speak(info.get('penyimpanan', ''))
else:
    st.info("Silakan unggah gambar obat atau ambil foto menggunakan kamera.")
