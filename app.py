import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
import gdown
from gtts import gTTS
import tempfile
from tensorflow.keras.preprocessing import image

# ========== KONFIGURASI GOOGLE DRIVE ========== #
GOOGLE_DRIVE_CONFIG = {
    "model_file_id": "1WEALsJVVZjTedzapj0ykmzVg3wf4-Yub",
    "dataset_file_id": "1V-HI64YbBUQmkd20IOqMzAEk88PlqECw",
    "model_filename": "model_obat_resnet152v2_100.h5",
    "dataset_filename": "dataset_obat.csv"
}

# ========== FUNGSI DOWNLOAD DARI DRIVE ========== #
def download_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        with st.spinner(f"Mengunduh {output_path} dari Google Drive..."):
            gdown.download(url, output_path, quiet=False)

# ========== KONFIGURASI STREAMLIT ========== #
st.set_page_config(page_title="ObatVision", layout="centered")
st.title("💊 ObatVision: Deteksi dan Info Obat")

# ========== LOAD MODEL & DATA ========== #
@st.cache_resource
def load_model():
    model_path = GOOGLE_DRIVE_CONFIG["model_filename"]
    download_from_drive(GOOGLE_DRIVE_CONFIG["model_file_id"], model_path)
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_data():
    dataset_path = GOOGLE_DRIVE_CONFIG["dataset_filename"]
    download_from_drive(GOOGLE_DRIVE_CONFIG["dataset_file_id"], dataset_path)
    return pd.read_csv(dataset_path)

model = load_model()
obat_info_df = load_data()
class_names = sorted(obat_info_df['label'].unique())

# ========== TEXT TO SPEECH (gTTS) ========== #
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# ========== INPUT GAMBAR ========== #
with st.sidebar:
    st.header("📷 Input Gambar")
    img_file = st.file_uploader("Unggah Gambar Obat", type=["jpg", "jpeg", "png"])
    camera_img = st.camera_input("Atau Ambil Gambar Realtime")

img_input = img_file if img_file else camera_img

# ========== PROSES PREDIKSI ========== #
if img_input:
    img = Image.open(img_input).convert('RGB')
    img_resized = img.resize((256, 256))
    
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # CARI INFO OBAT
    if predicted_label in obat_info_df['label'].values:
        info = obat_info_df[obat_info_df['label'] == predicted_label].iloc[0]

        # TAMPILAN UTAMA
        st.image(img, caption=f"Prediksi: {predicted_label} ({confidence:.2f}%)", use_column_width=True)
        st.subheader(f"💊 {info['nama_obat']}")
        st.markdown(f"""
        **Golongan:** {info['golongan']}  
        **Jenis:** {info['jenis']}  
        
        **Manfaat:** {info['manfaat']}  
        **Aturan Minum:** {info['aturan_minum']}  
        **Catatan:** {info['catatan']}  
        
        **🎯 Akurasi Prediksi:** {confidence:.2f}%
        """)

        # PERINGATAN + TTS
        st.warning("⚠ Aturan minum dapat berbeda-beda pada setiap orang. Ikuti saran dokter yang memahami kondisi Anda.")
        speak_text = f"Obat yang terdeteksi adalah {info['nama_obat']}. Aturan minum: {info['aturan_minum']}. Catatan: {info['catatan']}"
        speak(speak_text)

        # ========== MENU INFO LANJUTAN ========== #
        with st.expander("📂 Lihat lebih lanjut"):
            if st.button("Efek Samping"):
                st.info(info.get('efek_samping', 'Tidak tersedia'))
                speak(info.get('efek_samping', ''))
            if st.button("Pantangan Makanan"):
                st.info(info.get('pantangan_makanan', 'Tidak tersedia'))
                speak(info.get('pantangan_makanan', ''))
            if st.button("Interaksi Negatif"):
                st.info(info.get('interaksi_negatif', 'Tidak tersedia'))
                speak(info.get('interaksi_negatif', ''))
            if st.button("Jika Lupa Minum?"):
                st.info(info.get('jika_lupa_minum', 'Tidak tersedia'))
                speak(info.get('jika_lupa_minum', ''))
            if st.button("Cara Penyimpanan"):
                st.info(info.get('penyimpanan', 'Tidak tersedia'))
                speak(info.get('penyimpanan', ''))
    else:
        st.error("Label hasil prediksi tidak ditemukan di database obat. Harap periksa kembali dataset.")
else:
    st.info("Silakan unggah gambar obat atau ambil foto menggunakan kamera.")
