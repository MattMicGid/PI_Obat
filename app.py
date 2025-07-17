import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import gdown
import io
import tempfile
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.utils import img_to_array
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    TF_AVAILABLE = True
except ImportError:
    st.error("TensorFlow tidak terinstall. Silakan coba lagi nanti.")
    TF_AVAILABLE = False

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
    
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1A5276;
        transform: translateY(-2px);
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
def download_from_gdrive(file_id, filename):
    """Download file dari Google Drive"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Coba download dengan gdown
        try:
            gdown.download(url, filename, quiet=False)
            return True
        except Exception as e1:
            st.warning(f"Metode pertama gagal: {e1}")
            
            # Alternatif download dengan requests
            try:
                import requests
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e2:
                st.error(f"Semua metode download gagal: {e2}")
                return False
                
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        return False

# ========== LOAD MODEL & DATA ==========
@st.cache_resource
def load_model():
    """Load model TensorFlow"""
    if not TF_AVAILABLE:
        return None
        
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
        st.success("✅ Model berhasil dimuat!")
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
        st.success("✅ Dataset berhasil dimuat!")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# ========== TTS DENGAN WEB SPEECH API ==========
def create_tts_button(text, button_text="🔊 Dengarkan", key_suffix=""):
    """Membuat button TTS dengan JavaScript"""
    if st.button(button_text, key=f"tts_{key_suffix}"):
        # Bersihkan teks
        clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
        
        # Inject JavaScript untuk TTS
        js_code = f"""
        <script>
            function speakText() {{
                if ('speechSynthesis' in window) {{
                    // Stop speech sebelumnya
                    speechSynthesis.cancel();
                    
                    const utterance = new SpeechSynthesisUtterance("{clean_text}");
                    utterance.lang = 'id-ID';
                    utterance.rate = 0.9;
                    utterance.pitch = 1;
                    utterance.volume = 1;
                    
                    speechSynthesis.speak(utterance);
                }} else {{
                    alert('Browser tidak mendukung Text-to-Speech');
                }}
            }}
            
            // Jalankan fungsi
            speakText();
        </script>
        """
        
        # Render JavaScript
        st.components.v1.html(js_code, height=0)
        st.success("🔊 Sedang memutar audio...")

# ========== MAIN APP ==========
def main():
    # Header
    st.markdown('<h1 class="main-header">💊 ObatVision: Deteksi dan Info Obat</h1>', unsafe_allow_html=True)
    
    # Cek ketersediaan TensorFlow
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow tidak tersedia. Aplikasi tidak dapat berjalan.")
        return
    
    # Load model dan data
    with st.spinner("🔄 Memuat model dan dataset..."):
        model = load_model()
        obat_info_df = load_data()
    
    if model is None or obat_info_df is None:
        st.error("❌ Gagal memuat model atau dataset. Silakan refresh halaman.")
        return
    
    # Ambil class names
    try:
        class_names = sorted(obat_info_df['label'].unique())
    except Exception as e:
        st.error(f"Error mengambil class names: {e}")
        return
    
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
            try:
                img = Image.open(img_input).convert('RGB')
                st.image(img, caption="Gambar Input", use_column_width=True)
            except Exception as e:
                st.error(f"Error memuat gambar: {e}")
                return
        
        with col2:
            # Proses prediksi
            with st.spinner("🔍 Menganalisis gambar..."):
                try:
                    # Preprocessing
                    img_resized = img.resize((224, 224))
                    img_array = img_to_array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediksi
                    prediction = model.predict(img_array, verbose=0)[0]
                    predicted_index = np.argmax(prediction)
                    predicted_label = class_names[predicted_index]
                    confidence = prediction[predicted_index] * 100
                    
                    # Tampilkan hasil prediksi
                    st.markdown(f'<div class="prediction-info">'
                              f'<h3>🎯 Hasil Prediksi</h3>'
                              f'<p><strong>{predicted_label}</strong></p>'
                              f'<p>Akurasi: {confidence:.1f}%</p>'
                              f'</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi: {str(e)}")
                    return
        
        # ========== INFO OBAT ==========
        try:
            # Ambil info dari CSV
            matching_rows = obat_info_df[obat_info_df['label'] == predicted_label]
            if matching_rows.empty:
                st.error(f"❌ Informasi untuk {predicted_label} tidak ditemukan dalam database.")
                return
            
            info = matching_rows.iloc[0]
            
            # Tampilkan info obat
            st.markdown(f'<div class="info-card">'
                       f'<h2 class="drug-name">💊 {info.get("nama_obat", "Tidak diketahui")}</h2>'
                       f'<p><strong>Golongan:</strong> {info.get("golongan", "Tidak tersedia")}</p>'
                       f'<p><strong>Jenis:</strong> {info.get("jenis", "Tidak tersedia")}</p>'
                       f'<p><strong>Manfaat:</strong> {info.get("manfaat", "Tidak tersedia")}</p>'
                       f'<p><strong>Aturan Minum:</strong> {info.get("aturan_minum", "Tidak tersedia")}</p>'
                       f'<p><strong>Catatan:</strong> {info.get("catatan", "Tidak tersedia")}</p>'
                       f'</div>', unsafe_allow_html=True)
            
            # Peringatan
            warning_text = """⚠️ Aturan minum dapat berbeda-beda pada setiap orang, harus mengikuti saran dari dokter yang sudah cek kondisi pasien. 
            Aplikasi ini hanya untuk referensi dan tidak menggantikan konsultasi medis profesional."""
            
            st.markdown(f'<div class="warning-box">{warning_text}</div>', unsafe_allow_html=True)
            
            # TTS untuk info utama
            main_speech = f"Obat yang terdeteksi adalah {info.get('nama_obat', 'tidak diketahui')}. {info.get('manfaat', '')}. Aturan minum: {info.get('aturan_minum', '')}. {info.get('catatan', '')}"
            create_tts_button(main_speech, "🔊 Dengarkan Info Utama", "main_info")
            
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
                               f'<h4>⚠️ Efek Samping {info.get("nama_obat", "")}</h4>'
                               f'<p>{efek_samping}</p>'
                               f'</div>', unsafe_allow_html=True)
                    create_tts_button(f"Efek samping dari {info.get('nama_obat', '')}: {efek_samping}", 
                                    "🔊 Dengarkan", "efek_samping")
                
                if st.button("🚫 Pantangan Makanan", key="pantangan"):
                    pantangan = info.get('pantangan_makanan', 'Informasi tidak tersedia dalam database')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>🚫 Pantangan Makanan</h4>'
                               f'<p>{pantangan}</p>'
                               f'</div>', unsafe_allow_html=True)
                    create_tts_button(f"Pantangan makanan: {pantangan}", 
                                    "🔊 Dengarkan", "pantangan")
            
            with col2:
                if st.button("💊 Interaksi Negatif", key="interaksi"):
                    interaksi = info.get('interaksi_negatif', 'Informasi tidak tersedia dalam database')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>💊 Interaksi Negatif</h4>'
                               f'<p>{interaksi}</p>'
                               f'</div>', unsafe_allow_html=True)
                    create_tts_button(f"Interaksi negatif: {interaksi}", 
                                    "🔊 Dengarkan", "interaksi")
                
                if st.button("❓ Jika Lupa Minum?", key="lupa_minum"):
                    lupa_minum = info.get('jika_lupa_minum', 'Segera minum jika baru ingat, kecuali sudah mendekati waktu dosis berikutnya')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>❓ Jika Lupa Minum Obat</h4>'
                               f'<p>{lupa_minum}</p>'
                               f'</div>', unsafe_allow_html=True)
                    create_tts_button(f"Jika lupa minum obat: {lupa_minum}", 
                                    "🔊 Dengarkan", "lupa_minum")
            
            with col3:
                if st.button("🏠 Cara Penyimpanan", key="penyimpanan"):
                    penyimpanan = info.get('penyimpanan', 'Simpan di tempat sejuk, kering, dan terhindar dari sinar matahari langsung')
                    st.markdown(f'<div class="popup-info">'
                               f'<h4>🏠 Cara Penyimpanan</h4>'
                               f'<p>{penyimpanan}</p>'
                               f'</div>', unsafe_allow_html=True)
                    create_tts_button(f"Cara penyimpanan: {penyimpanan}", 
                                    "🔊 Dengarkan", "penyimpanan")
            
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