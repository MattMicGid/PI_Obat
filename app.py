import streamlit as st
from PIL import Image
import pandas as pd
from gtts import gTTS
import tempfile
import base64
import gdown
import os
import random

# ========== KONFIG ========== #
st.set_page_config(page_title="💊 ObatVision", layout="centered")
st.title("💊 ObatVision - Dummy Versi GDrive")

GOOGLE_DRIVE_CONFIG = {
    "dataset_file_id": "1V-HI64YbBUQmkd20IOqMzAEk88PlqECw",
    "dataset_filename": "dataset_obat.csv"
}

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

download_from_drive(GOOGLE_DRIVE_CONFIG["dataset_file_id"], GOOGLE_DRIVE_CONFIG["dataset_filename"])

# ========== LOAD CSV ========== #
@st.cache_data
def load_data():
    return pd.read_csv(GOOGLE_DRIVE_CONFIG["dataset_filename"])

dataset = load_data()

# ========== UTIL AUDIO ========== #
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

# ========== INPUT GAMBAR ========== #
st.subheader("Unggah atau ambil gambar obat:")
img_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
camera_img = st.camera_input("Atau Ambil Gambar Lewat Kamera")

input_image = None
if img_file:
    input_image = Image.open(img_file)
elif camera_img:
    input_image = Image.open(camera_img)

# ========== SIMULASI OUTPUT ========== #
if input_image:
    st.image(input_image, caption="Gambar Obat", use_column_width=True)
    
    row = dataset.sample(1).iloc[0]
    nama_obat = row["nama_obat"]
    confidence = random.uniform(85, 99)

    st.success(f"✅ Prediksi: **{nama_obat}** ({confidence:.2f}% confidence)")

    st.markdown(f"""
    **Golongan:** {row['golongan']}  
    **Jenis:** {row['jenis']}  
    **Manfaat:** {row['manfaat']}  
    **Aturan Minum:** {row['aturan_minum']}  
    **Catatan:** {row['catatan']}
    """)

    peringatan = "Aturan minum dapat berbeda-beda pada setiap orang, harus mengikuti saran dari dokter yang sudah cek kondisi pasien."
    st.warning(peringatan)
    play_audio(peringatan)

    st.markdown("### Lihat lebih lanjut:")
    show_info_popup("📌 Efek Samping", row["efek_samping"])
    show_info_popup("🥦 Pantangan Makanan", row["pantangan_makanan"])
    show_info_popup("❗ Interaksi Negatif", row["interaksi_negatif"])
    show_info_popup("⏰ Jika Lupa Minum", row["jika_lupa_minum"])
    show_info_popup("💾 Cara Penyimpanan", row["penyimpanan"])
