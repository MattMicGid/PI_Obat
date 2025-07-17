import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Gambar dengan TensorFlow",
    page_icon="🖼️",
    layout="wide"
)

# Cache model untuk performa yang lebih baik
@st.cache_resource
def load_model():
    """Load pre-trained MobileNetV2 model"""
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(img):
    """Preprocess gambar untuk prediksi"""
    # Resize gambar ke 224x224
    img = img.resize((224, 224))
    # Convert ke array
    img_array = image.img_to_array(img)
    # Expand dimensi
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_array):
    """Prediksi gambar"""
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def main():
    st.title("🖼️ Klasifikasi Gambar dengan TensorFlow")
    st.markdown("Upload gambar dan lihat prediksi dari model MobileNetV2!")
    
    # Sidebar info
    st.sidebar.header("ℹ️ Info Aplikasi")
    st.sidebar.markdown("""
    **Model:** MobileNetV2 (ImageNet)
    **Framework:** TensorFlow 2.x
    **Deployment:** Streamlit Cloud
    
    **Cara Penggunaan:**
    1. Upload gambar (JPG, JPEG, PNG)
    2. Tunggu proses prediksi
    3. Lihat hasil klasifikasi
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    st.success("Model berhasil dimuat!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar dalam format JPG, JPEG, atau PNG"
    )
    
    if uploaded_file is not None:
        # Tampilkan gambar
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Gambar Input")
            # Baca gambar
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang diupload", use_column_width=True)
            
            # Info gambar
            st.write(f"**Ukuran:** {img.size}")
            st.write(f"**Mode:** {img.mode}")
        
        with col2:
            st.subheader("🤖 Hasil Prediksi")
            
            # Preprocess dan prediksi
            with st.spinner("Memproses gambar..."):
                # Preprocess
                img_array = preprocess_image(img)
                
                # Prediksi
                predictions = predict_image(model, img_array)
            
            # Tampilkan hasil
            st.write("**Top 3 Prediksi:**")
            
            for i, (imagenet_id, label, score) in enumerate(predictions):
                confidence = score * 100
                
                # Progress bar untuk confidence
                st.write(f"**{i+1}. {label}**")
                st.progress(confidence / 100)
                st.write(f"Confidence: {confidence:.2f}%")
                st.write("---")
            
            # Prediksi terbaik
            best_prediction = predictions[0]
            st.success(f"**Prediksi Terbaik:** {best_prediction[1]} ({best_prediction[2]*100:.2f}%)")
    
    # Footer
    st.markdown("---")
    st.markdown("**Dibuat dengan ❤️ menggunakan Streamlit dan TensorFlow**")

if __name__ == "__main__":
    main()