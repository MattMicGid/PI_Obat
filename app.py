import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from streamlit_extras.stylable_container import stylable_container
import base64
import io
import requests
import tempfile
import gdown

# ========== CONFIGURASI ==========
st.set_page_config(page_title="ObatVision", layout="centered")
st.title("💊 ObatVision: Deteksi dan Info Obat")

# ========== GOOGLE DRIVE CONFIGURATION ==========
# Ganti dengan Google Drive file IDs Anda
GOOGLE_DRIVE_CONFIG = {
    "model_file_id": "1WEALsJVVZjTedzapj0ykmzVg3wf4-Yub",  # Ganti dengan file ID model Anda
    "dataset_file_id": "1V-HI64YbBUQmkd20IOqMzAEk88PlqECw",  # Ganti dengan file ID dataset Anda
    "model_filename": "modal_obat_1.h5",
    "dataset_filename": "dataset_obat.csv"
}

# ========== FUNGSI DOWNLOAD DARI GOOGLE DRIVE ==========

def get_google_drive_download_url(file_id):
    """Convert Google Drive file ID to direct download URL"""
    return f"https://drive.google.com/uc?id={file_id}&export=download"

def download_from_gdrive_gdown(file_id, output_path):
    """Download file from Google Drive using gdown"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error downloading with gdown: {str(e)}")
        return False

def download_from_gdrive_requests(file_id, output_path):
    """Download file from Google Drive using requests"""
    try:
        url = get_google_drive_download_url(file_id)
        
        with requests.Session() as session:
            response = session.get(url, stream=True)
            
            # Handle large files with confirmation token
            if "download_warning" in response.text:
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'export': 'download', 'confirm': value}
                        response = session.get(url, params=params, stream=True)
                        break
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                st.error(f"Failed to download: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        st.error(f"Error downloading with requests: {str(e)}")
        return False

def download_file_from_gdrive(file_id, filename):
    """Download file from Google Drive with multiple methods"""
    
    # Create temp directory if not exists
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        return file_path
    
    st.info(f"Downloading {filename} from Google Drive...")
    
    # Progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Try gdown first (recommended for large files)
    progress_text.text("Trying gdown method...")
    progress_bar.progress(25)
    
    if download_from_gdrive_gdown(file_id, file_path):
        progress_bar.progress(100)
        progress_text.text("Download completed!")
        return file_path
    
    # If gdown fails, try requests method
    progress_text.text("Trying requests method...")
    progress_bar.progress(50)
    
    if download_from_gdrive_requests(file_id, file_path):
        progress_bar.progress(100)
        progress_text.text("Download completed!")
        return file_path
    
    # If both methods fail
    progress_bar.progress(100)
    progress_text.text("Download failed!")
    st.error("Could not download file from Google Drive")
    return None

# ========== LOAD MODEL & DATA WITH CACHING ==========

@st.cache_resource
def load_model():
    """Load model from Google Drive"""
    try:
        model_path = download_file_from_gdrive(
            GOOGLE_DRIVE_CONFIG["model_file_id"], 
            GOOGLE_DRIVE_CONFIG["model_filename"]
        )
        
        if model_path and os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            st.success("✅ Model loaded successfully from Google Drive!")
            return model
        else:
            st.error("❌ Failed to load model from Google Drive")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load dataset from Google Drive"""
    try:
        dataset_path = download_file_from_gdrive(
            GOOGLE_DRIVE_CONFIG["dataset_file_id"], 
            GOOGLE_DRIVE_CONFIG["dataset_filename"]
        )
        
        if dataset_path and os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            st.success("✅ Dataset loaded successfully from Google Drive!")
            return df
        else:
            st.error("❌ Failed to load dataset from Google Drive")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

# ========== CONFIGURATION SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Google Drive file configuration
    st.subheader("📁 Google Drive Files")
    
    with st.expander("🔧 Edit Google Drive IDs"):
        new_model_id = st.text_input("Model File ID", value=GOOGLE_DRIVE_CONFIG["model_file_id"])
        new_dataset_id = st.text_input("Dataset File ID", value=GOOGLE_DRIVE_CONFIG["dataset_file_id"])
        
        if st.button("Update File IDs"):
            GOOGLE_DRIVE_CONFIG["model_file_id"] = new_model_id
            GOOGLE_DRIVE_CONFIG["dataset_file_id"] = new_dataset_id
            st.success("File IDs updated! Please restart the app.")
    
    # How to get Google Drive file ID
    st.info("""
    **How to get Google Drive file ID:**
    1. Upload your file to Google Drive
    2. Right-click → Get link
    3. Make sure it's set to "Anyone with the link can view"
    4. Copy the ID from the URL:
    `https://drive.google.com/file/d/FILE_ID_HERE/view`
    """)

# Load model and data
model = load_model()
obat_info_df = load_data()

# Check if both model and data are loaded
if model is None or obat_info_df is None:
    st.error("❌ Cannot proceed without model and dataset. Please check your Google Drive configuration.")
    st.stop()

class_names = sorted(obat_info_df['label'].unique())

# ========== TEXT TO SPEECH OPTIONS ==========

def create_audio_html(text, lang="id-ID"):
    """Create HTML audio player using browser's speech synthesis"""
    html_string = f"""
    <div style="margin: 10px 0;">
        <button onclick="speakText()" style="
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        ">🔊 Putar Audio</button>
    </div>
    <script>
        function speakText() {{
            const text = `{text}`;
            const speech = new SpeechSynthesisUtterance(text);
            speech.lang = '{lang}';
            speech.rate = 0.8;
            speech.pitch = 1;
            speech.volume = 1;
            
            const voices = speechSynthesis.getVoices();
            const indonesianVoice = voices.find(voice => 
                voice.lang.includes('id') || voice.lang.includes('ID')
            );
            if (indonesianVoice) {{
                speech.voice = indonesianVoice;
            }}
            
            speechSynthesis.speak(speech);
        }}
        
        if (speechSynthesis.onvoiceschanged !== undefined) {{
            speechSynthesis.onvoiceschanged = function() {{}};
        }}
    </script>
    """
    return html_string

def create_gtts_audio(text, lang="id"):
    """Create audio using Google Text-to-Speech"""
    try:
        from gtts import gTTS
        
        tts = gTTS(text=text, lang=lang, slow=False)
        
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_bytes = fp.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        html_string = f"""
        <audio controls autoplay style="width: 100%; margin: 10px 0;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Browser Anda tidak mendukung audio.
        </audio>
        """
        return html_string
    except ImportError:
        st.error("gTTS tidak terinstall. Jalankan: pip install gtts")
        return None
    except Exception as e:
        st.error(f"Error gTTS: {str(e)}")
        return None

def create_streamlit_audio(text):
    """Create downloadable audio info"""
    st.info(f"🔊 Teks untuk dibaca: {text}")
    st.caption("Gunakan screen reader atau baca teks di atas")

# Choose TTS method
TTS_METHOD = st.sidebar.selectbox(
    "Pilih Metode Audio:",
    ["Browser TTS", "Google TTS", "Text Only"]
)

def play_audio(text):
    """Play audio based on selected method"""
    if TTS_METHOD == "Browser TTS":
        st.components.v1.html(create_audio_html(text), height=60)
    elif TTS_METHOD == "Google TTS":
        audio_html = create_gtts_audio(text)
        if audio_html:
            st.components.v1.html(audio_html, height=60)
        else:
            create_streamlit_audio(text)
    else:
        create_streamlit_audio(text)

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

    # Prediction
    with st.spinner("Memproses prediksi..."):
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

    # Audio untuk info utama
    main_text = f"Obat yang terdeteksi adalah {info['nama_obat']}. Aturan minum: {info['aturan_minum']}. Perhatian: {info['catatan']}"
    play_audio(main_text)

    # Peringatan
    st.warning("⚠ Aturan minum dapat berbeda-beda pada setiap orang. Ikuti saran dokter yang memahami kondisi Anda.")

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
                efek_samping = info.get('efek_samping', 'Tidak tersedia')
                st.info(f"Efek samping dari {info['nama_obat']}: {efek_samping}")
                play_audio(f"Efek samping: {efek_samping}")
                
            if st.button("Pantangan Makanan"):
                pantangan = info.get('pantangan_makanan', 'Tidak tersedia')
                st.info(pantangan)
                play_audio(f"Pantangan makanan: {pantangan}")
                
            if st.button("Interaksi Negatif"):
                interaksi = info.get('interaksi_negatif', 'Tidak tersedia')
                st.info(interaksi)
                play_audio(f"Interaksi negatif: {interaksi}")
                
        with col2:
            if st.button("Jika Lupa Minum?"):
                lupa_minum = info.get('jika_lupa_minum', 'Tidak tersedia')
                st.info(lupa_minum)
                play_audio(f"Jika lupa minum: {lupa_minum}")
                
            if st.button("Cara Penyimpanan"):
                penyimpanan = info.get('penyimpanan', 'Tidak tersedia')
                st.info(penyimpanan)
                play_audio(f"Cara penyimpanan: {penyimpanan}")
                
else:
    st.info("Silakan unggah gambar obat atau ambil foto menggunakan kamera.")

# ========== FOOTER INFO ==========
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Dependencies")
st.sidebar.code("""
streamlit
tensorflow
pillow
pandas
numpy
streamlit-extras
gtts
gdown
requests
""")

st.sidebar.markdown("### 🔧 Setup Google Drive")
st.sidebar.markdown("""
1. Upload model (.h5) dan dataset (.csv) ke Google Drive
2. Ubah sharing setting menjadi "Anyone with the link can view"
3. Copy file ID dari URL Google Drive
4. Paste ke konfigurasi di atas
""")