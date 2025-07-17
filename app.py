import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import threading
import time
import gdown
from streamlit_option_menu import option_menu

# ========== CONFIGURASI ==========
st.set_page_config(
    page_title="ObatVision", 
    layout="centered",
    page_icon="💊",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .menu-button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
        width: 100%;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>💊 ObatVision: Deteksi dan Info Obat</h1>", unsafe_allow_html=True)

# ========== LOAD MODEL & DATA ==========
@st.cache_resource
def load_model():
    """Load model dari Google Drive jika tidak ada di local"""
    model_path = "model_obat_1.h5"
    
    if not os.path.exists(model_path):
        # Jika model tidak ada, download dari Google Drive
        st.info("📥 Mengunduh model... (hanya sekali)")
        # Ganti dengan Google Drive file ID Anda
        file_id = "1WEALsJVVZjTedzapj0ykmzVg3wf4-Yub"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Load dataset obat"""
    try:
        df = pd.read_csv("dataset_obat.csv")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# ========== TEXT TO SPEECH (Web-based alternative) ==========
def speak_text(text):
    """Text to speech menggunakan browser API"""
    js_code = f"""
    <script>
        function speakText() {{
            if ('speechSynthesis' in window) {{
                const utterance = new SpeechSynthesisUtterance("{text}");
                utterance.lang = 'id-ID';
                utterance.rate = 0.8;
                utterance.pitch = 1;
                speechSynthesis.speak(utterance);
            }} else {{
                console.log("Speech synthesis not supported");
            }}
        }}
        speakText();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# ========== FUNGSI PREDIKSI ==========
def predict_image(img, model, class_names):
    """Prediksi gambar obat"""
    try:
        # Preprocessing
        img_resized = img.resize((256, 256))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediksi
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = prediction[predicted_index] * 100
        
        return predicted_label, confidence
    except Exception as e:
        st.error(f"❌ Error dalam prediksi: {e}")
        return None, 0

# ========== FUNGSI KAMERA REALTIME ==========
def capture_camera():
    """Capture gambar dari kamera dengan preprocessing"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Tidak dapat mengakses kamera")
        return None
    
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame untuk konsistensi
        frame_resized = cv2.resize(frame, (256, 256))
        
        # Tampilkan frame
        stframe.image(frame_resized, channels="BGR", use_column_width=True)
        
        # Tombol capture
        if st.button("📸 Ambil Foto"):
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            cap.release()
            return img
    
    cap.release()
    return None

# ========== MAIN APP ==========
def main():
    # Load model dan data
    model = load_model()
    obat_info_df = load_data()
    
    if model is None or obat_info_df is None:
        st.error("❌ Gagal memuat model atau dataset. Silakan coba lagi.")
        return
    
    class_names = sorted(obat_info_df['label'].unique())
    
    # Sidebar untuk input
    with st.sidebar:
        st.header("📷 Input Gambar")
        
        # Menu pilihan
        input_method = option_menu(
            menu_title="Pilih Metode Input",
            options=["📁 Upload Gambar", "📸 Kamera Realtime"],
            icons=["cloud-upload", "camera"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )
        
        img_input = None
        
        if input_method == "📁 Upload Gambar":
            img_file = st.file_uploader(
                "Unggah Gambar Obat", 
                type=["jpg", "jpeg", "png"],
                help="Pilih gambar obat yang ingin diidentifikasi"
            )
            if img_file:
                img_input = Image.open(img_file).convert('RGB')
        
        elif input_method == "📸 Kamera Realtime":
            st.info("📱 Gunakan kamera untuk mengambil foto obat")
            camera_img = st.camera_input("Ambil Foto Obat")
            if camera_img:
                img_input = Image.open(camera_img).convert('RGB')
    
    # ========== PROSES PREDIKSI ==========
    if img_input:
        # Prediksi
        predicted_label, confidence = predict_image(img_input, model, class_names)
        
        if predicted_label:
            # Ambil info dari CSV
            info = obat_info_df[obat_info_df['label'] == predicted_label].iloc[0]
            
            # Tampilkan gambar dengan hasil prediksi
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img_input, caption="Gambar Input", use_column_width=True)
            
            with col2:
                # Confidence styling
                if confidence >= 80:
                    conf_class = "confidence-high"
                elif confidence >= 60:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"
                
                st.markdown(f"""
                <div class="prediction-container">
                    <h3>🎯 Hasil Prediksi</h3>
                    <h2>{predicted_label}</h2>
                    <p class="{conf_class}">Akurasi: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Info obat
            st.markdown(f"""
            <div class="info-box">
                <h3>💊 {info['nama_obat']}</h3>
                <p><strong>Golongan:</strong> {info['golongan']}</p>
                <p><strong>Jenis:</strong> {info['jenis']}</p>
                <p><strong>Manfaat:</strong> {info['manfaat']}</p>
                <p><strong>Aturan Minum:</strong> {info['aturan_minum']}</p>
                <p><strong>Catatan:</strong> {info['catatan']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Peringatan
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Peringatan Penting</h4>
                <p>Aturan minum dapat berbeda-beda pada setiap orang, harus mengikuti saran dari dokter yang sudah cek kondisi pasien. 
                Solusi: Konsultasikan dengan dokter untuk koreksi jadwal minum obat yang sesuai dengan kondisi Anda.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Text to speech
            speech_text = f"Obat yang terdeteksi adalah {info['nama_obat']}. Aturan minum: {info['aturan_minum']}. Catatan: {info['catatan']}"
            
            if st.button("🔊 Dengarkan Info Obat"):
                speak_text(speech_text)
            
            # ========== MENU LANJUTAN ==========
            st.markdown("### 📂 Informasi Tambahan")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💊 Efek Samping"):
                    with st.expander("Efek Samping", expanded=True):
                        efek_samping = info.get('efek_samping', 'Informasi tidak tersedia')
                        st.write(efek_samping)
                        speak_text(f"Efek samping: {efek_samping}")
                
                if st.button("🥗 Pantangan Makanan"):
                    with st.expander("Pantangan Makanan", expanded=True):
                        pantangan = info.get('pantangan_makanan', 'Informasi tidak tersedia')
                        st.write(pantangan)
                        speak_text(f"Pantangan makanan: {pantangan}")
            
            with col2:
                if st.button("⚠️ Interaksi Negatif"):
                    with st.expander("Interaksi Negatif", expanded=True):
                        interaksi = info.get('interaksi_negatif', 'Informasi tidak tersedia')
                        st.write(interaksi)
                        speak_text(f"Interaksi negatif: {interaksi}")
                
                if st.button("❓ Jika Lupa Minum?"):
                    with st.expander("Jika Lupa Minum", expanded=True):
                        lupa_minum = info.get('jika_lupa_minum', 'Informasi tidak tersedia')
                        st.write(lupa_minum)
                        speak_text(f"Jika lupa minum: {lupa_minum}")
            
            with col3:
                if st.button("🏠 Cara Penyimpanan"):
                    with st.expander("Cara Penyimpanan", expanded=True):
                        penyimpanan = info.get('penyimpanan', 'Informasi tidak tersedia')
                        st.write(penyimpanan)
                        speak_text(f"Cara penyimpanan: {penyimpanan}")
                
                if st.button("📋 Info Lengkap"):
                    with st.expander("Informasi Lengkap", expanded=True):
                        for col in info.index:
                            if col not in ['label']:
                                st.write(f"**{col.replace('_', ' ').title()}:** {info[col]}")
    
    else:
        st.info("📌 Silakan unggah gambar obat atau ambil foto menggunakan kamera untuk memulai identifikasi.")
        
        # Tampilkan contoh penggunaan
        st.markdown("""
        ### 🎯 Cara Penggunaan:
        1. **Upload Gambar**: Pilih foto obat dari galeri Anda
        2. **Kamera Realtime**: Ambil foto langsung menggunakan kamera
        3. **Lihat Hasil**: Dapatkan informasi lengkap tentang obat
        4. **Dengarkan**: Aktifkan text-to-speech untuk mendengar informasi
        
        ### 📝 Tips untuk Hasil Terbaik:
        - Pastikan gambar obat jelas dan tidak buram
        - Cahaya yang cukup untuk foto yang baik
        - Posisikan obat di tengah frame
        - Hindari bayangan atau pantulan cahaya
        """)

if __name__ == "__main__":
    main()