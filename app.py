import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gdown
import io
import tempfile
import requests
import time

# ========== CONFIGURASI ==========
st.set_page_config(
    page_title="ObatVision", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        color: #856404;
    }
    
    .info-card {
        background-color: #F8F9FA;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .popup-info {
        background-color: #E3F2FD;
        border: 1px solid #90CAF9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #0D47A1;
    }
    
    .confidence-badge {
        background-color: #4CAF50;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    
    .button-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin: 20px 0;
    }
    
    .custom-button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .custom-button:hover {
        background-color: #1A5276;
        transform: translateY(-2px);
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .drug-name {
        color: #2E86AB;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .prediction-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== KONFIGURASI GOOGLE DRIVE ==========
GOOGLE_DRIVE_CONFIG = {
    "model_file_id": "1WEALsJVVZjTedzapj0ykmzVg3wf4-Yub",
    "dataset_file_id": "1V-HI64YbBUQmkd20IOqMzAEk88PlqECw",
    "model_filename": "model_obat_resnet152v2_100.h5",
    "dataset_filename": "dataset_obat.csv"
}

# ========== FUNGSI DOWNLOAD ==========
@st.cache_data
def download_from_gdrive(file_id, filename):
    """Download file dari Google Drive"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        return False

# ========== LOAD MODEL & DATA ==========
@st.cache_resource
def load_model():
    """Load model TensorFlow"""
    model_path = GOOGLE_DRIVE_CONFIG["model_filename"]
    
    if not os.path.exists(model_path):
        with st.spinner("🔄 Mengunduh model... Mohon tunggu sebentar..."):
            success = download_from_gdrive(
                GOOGLE_DRIVE_CONFIG["model_file_id"], 
                model_path
            )
            if not success:
                st.error("Gagal mengunduh model!")
                return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load dataset obat"""
    dataset_path = GOOGLE_DRIVE_CONFIG["dataset_filename"]
    
    if not os.path.exists(dataset_path):
        with st.spinner("📊 Mengunduh dataset... Mohon tunggu sebentar..."):
            success = download_from_gdrive(
                GOOGLE_DRIVE_CONFIG["dataset_file_id"], 
                dataset_path
            )
            if not success:
                st.error("Gagal mengunduh dataset!")
                return None
    
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# ========== TTS DENGAN WEB SPEECH API ==========
def speak_text(text, key_suffix=""):
    """Text to Speech menggunakan JavaScript"""
    # Membersihkan teks dari karakter khusus
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
    
    # Buat button untuk memicu TTS
    if st.button(f"🔊 Dengarkan", key=f"tts_{key_suffix}"):
        js_code = f"""
        <script>
            function speakText() {{
                if ('speechSynthesis' in window) {{
                    const utterance = new SpeechSynthesisUtterance("{clean_text}");
                    utterance.lang = 'id-ID';
                    utterance.rate = 0.9;
                    utterance.pitch = 1;
                    speechSynthesis.speak(utterance);
                }} else {{
                    alert('Browser tidak mendukung Text-to-Speech');
                }}
            }}
            speakText();
        </script>
        """
        st.components.v1.html(js_code, height=0)

# ========== MAIN APP ==========
def main():
    # Header
    st.markdown('<h1 class="main-header">💊 ObatVision: Deteksi dan Info Obat</h1>', unsafe_allow_html=True)
    
    # Load model dan data
    model = load_model()
    obat_info_df = load_data()
    
    if model is None or obat_info_df is None:
        st.error("❌ Gagal memuat model atau dataset. Silakan refresh halaman.")
        return
    
    # Ambil class names
    class_names = sorted(obat_info_df['label'].unique())
    
    # ========== INPUT GAMBAR ==========
    with st.sidebar:
        st.header("📷 Input Gambar")
        st.markdown("Pilih salah satu metode input:")
        
        # Upload file
        img_file = st.file_uploader(
            "📁 Unggah Gambar Obat", 
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        st.markdown("---")
        
        # Camera input
        camera_img = st.camera_input(
            "📸 Ambil Gambar Realtime",
            help="Pastikan gambar obat terlihat jelas"
        )
        
        # Pilih input yang digunakan
        img_input = img_file if img_file else camera_img
        
        if img_input:
            st.success("✅ Gambar berhasil dimuat!")
    
    # ========== PROSES & PREDIKSI ==========
    if img_input:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Tampilkan gambar
            img = Image.open(img_input).convert('RGB')
            st.image(img, caption="Gambar Input", use_column_width=True)
        
        with col2:
            # Proses prediksi
            with st.spinner("🔍 Menganalisis gambar..."):
                try:
                    # Preprocessing
                    img_resized = img.resize((224, 224))  # Sesuaikan dengan input model
                    img_array = image.img_to_array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediksi
                    prediction = model.predict(img_array)[0]
                    predicted_index = np.argmax(prediction)
                    predicted_label = class_names[predicted_index]
                    confidence = prediction[predicted_index] * 100
                    
                    # Tampilkan hasil prediksi
                    st.markdown(f'<div class="prediction-info">'
                              f'<h3>🎯 Hasil Prediksi</h3>'
                              f'<p><strong>{predicted_label}</strong></p>'
                              f'<p>Akurasi: {confidence:.2f}%</p>'
                              f'</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi: {str(e)}")
                    return
        
        # ========== INFO OBAT ==========
        try:
            # Ambil info dari CSV
            info = obat_info_df[obat_info_df['label'] == predicted_label].iloc[0]
            
            # Tampilkan info obat
            st.markdown(f'<div class="info-card">'
                       f'<h2 class="drug-name">💊 {info["nama_obat"]}</h2>'
                       f'<p><strong>Golongan:</strong> {info["golongan"]}</p>'
                       f'<p><strong>Jenis:</strong> {info["jenis"]}</p>'
                       f'<p><strong>Manfaat:</strong> {info["manfaat"]}</p>'
                       f'<p><strong>Aturan Minum:</strong> {info["aturan_minum"]}</p>'
                       f'<p><strong>Catatan:</strong> {info["catatan"]}</p>'
                       f'</div>', unsafe_allow_html=True)
            
            # Peringatan
            warning_text = """⚠️ Aturan minum dapat berbeda-beda pada setiap orang, harus mengikuti saran dari dokter yang sudah cek kondisi pasien. 
            Aplikasi ini hanya untuk referensi dan tidak menggantikan konsultasi medis profesional."""
            
            st.markdown(f'<div class="warning-box">{warning_text}</div>', unsafe_allow_html=True)
            
            # TTS untuk info utama
            main_speech = f"Obat yang terdeteksi adalah {info['nama_obat']}. {info['manfaat']}. Aturan minum: {info['aturan_minum']}. {info['catatan']}"
            speak_text(main_speech, "main_info")
            
            # ========== MENU LANJUTAN ==========
            st.markdown("---")
            st.markdown("### 📂 Informasi Lebih Lanjut")
            st.markdown("Klik tombol di bawah untuk mendapatkan informasi detail:")
            
            # Buat columns untuk layout button
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⚠️ Efek Samping", key="efek_samping"):
                    efek_samping = info.get('efek_samping', 'Informasi tidak tersedia dalam database')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>⚠️ Efek Samping {info["nama_obat"]}</h4>'
                               f'<p>{efek_samping}</p>'
                               f'</div>', unsafe_allow_html=True)
                    speak_text(f"Efek samping dari {info['nama_obat']}: {efek_samping}", "efek_samping")
                
                if st.button("🚫 Pantangan Makanan", key="pantangan"):
                    pantangan = info.get('pantangan_makanan', 'Informasi tidak tersedia dalam database')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>🚫 Pantangan Makanan</h4>'
                               f'<p>{pantangan}</p>'
                               f'</div>', unsafe_allow_html=True)
                    speak_text(f"Pantangan makanan: {pantangan}", "pantangan")
            
            with col2:
                if st.button("💊 Interaksi Negatif", key="interaksi"):
                    interaksi = info.get('interaksi_negatif', 'Informasi tidak tersedia dalam database')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>💊 Interaksi Negatif</h4>'
                               f'<p>{interaksi}</p>'
                               f'</div>', unsafe_allow_html=True)
                    speak_text(f"Interaksi negatif: {interaksi}", "interaksi")
                
                if st.button("❓ Jika Lupa Minum?", key="lupa_minum"):
                    lupa_minum = info.get('jika_lupa_minum', 'Segera minum jika baru ingat, kecuali sudah mendekati waktu dosis berikutnya')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>❓ Jika Lupa Minum Obat</h4>'
                               f'<p>{lupa_minum}</p>'
                               f'</div>', unsafe_allow_html=True)
                    speak_text(f"Jika lupa minum obat: {lupa_minum}", "lupa_minum")
            
            with col3:
                if st.button("🏠 Cara Penyimpanan", key="penyimpanan"):
                    penyimpanan = info.get('penyimpanan', 'Simpan di tempat sejuk, kering, dan terhindar dari sinar matahari langsung')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>🏠 Cara Penyimpanan</h4>'
                               f'<p>{penyimpanan}</p>'
                               f'</div>', unsafe_allow_html=True)
                    speak_text(f"Cara penyimpanan: {penyimpanan}", "penyimpanan")
            
            # Session state untuk menyimpan informasi yang sudah diklik
            if 'clicked_buttons' not in st.session_state:
                st.session_state.clicked_buttons = []
            
        except Exception as e:
            st.error(f"❌ Error mengambil informasi obat: {str(e)}")
    
    else:
        # Tampilan awal
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h3>🔍 Cara Menggunakan ObatVision</h3>
            <p>1. Pilih metode input gambar di sidebar</p>
            <p>2. Upload gambar obat atau ambil foto langsung</p>
            <p>3. Tunggu proses analisis</p>
            <p>4. Dapatkan informasi lengkap tentang obat</p>
            <br>
            <p><em>Silakan unggah gambar obat atau ambil foto menggunakan kamera untuk memulai.</em></p>
        </div>
        """, unsafe_allow_html=True)

# ========== FOOTER ==========
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>💊 ObatVision v1.0 | Deteksi Obat dengan AI</p>
        <p><small>⚠️ Aplikasi ini hanya untuk referensi. Selalu konsultasikan dengan dokter atau apoteker untuk informasi medis yang akurat.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()