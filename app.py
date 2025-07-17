import streamlit as st
from PIL import Image
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import speech_recognition as sr
import pyttsx3
import cv2
import mediapipe as mp
import tensorflow as tf
import threading

# Page config
st.set_page_config(page_title="AI SignLang Tutor", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧏 AI SignLang Tutor</h1>", unsafe_allow_html=True)

# Banner Image
if os.path.exists("image.jpg"):
    st.image("image.jpg", use_container_width=True)

# Motivational Quote
st.markdown("""
<p style='font-size:20px; text-align:center; font-weight:bold;'>
💪 Physical disabilities are not barriers 🚫 to success 🌟<br>
They're just different paths to greatness ✨🚀
</p>
<hr style='border:none; height:2px; background:#ccc;'>
""", unsafe_allow_html=True)

# Extra space before tabs
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# 💅 Tab Styling
st.markdown("""
<style>
div[data-baseweb="tab-list"] button {
    font-size: 18px !important;
    margin-right: 25px !important;
    padding: 10px 20px !important;
    border-radius: 8px;
    background-color: #f5f5f5;
    transition: all 0.3s ease;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    font-weight: bold;
    background-color: #d0ebff !important;
    color: #004080 !important;
}
div[data-baseweb="tab-list"] button:hover {
    background-color: #e6f4ff;
}
</style>
""", unsafe_allow_html=True)

# --- Voice Output ---
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        st.warning("⚠️ Voice output engine is already active. Try again after refresh.")

# --- Helpers ---
def record_audio(duration=5, fs=44100):
    st.info("🎙️ Listening... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio, fs

def save_wav(filename, audio, fs):
    scipy.io.wavfile.write(filename, fs, audio)

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "❌ Could not understand audio."
        except sr.RequestError as e:
            return f"❗ API error: {e}"

def get_keywords(sentence):
    return sentence.lower().split()

def ask_amazon_q(prompt):
    prompt = prompt.lower().strip()
    q_responses = {
    # Alphabets (sample A–E)
    "how to sign A": "Fist with thumb on the side.",
    "how to sign B": "Palm out, fingers together, thumb across palm.",
    "how to sign C": "Make a C-shape with hand.",
    "how to sign D": "Index finger up, touch thumb and middle.",
    "how to sign E": "Fingertips touch thumb, palm forward.",

    # Greetings & Emotions
    "how to sign hello": "Salute starting from the forehead.",
    "how to sign thank you": "Fingers at chin moving forward.",
    "how to sign sorry": "Rub fist over chest in circular motion.",
    "how to sign please": "Flat hand on chest, move in circular motion.",
    "how to sign good morning": "Hand at chin moves forward then sun rising motion.",
    "how to sign good night": "Palm forward, then hands fold like sleeping.",
    "how to sign welcome": "Both hands move inward toward the chest.",
    "how to sign bye": "Wave your hand as you would naturally.",
    "how to sign happy": "Flat hands circle up from chest outward with smile.",
    "how to sign sad": "Open hands move down past the face.",
    "how to sign angry": "Claw hands pull down from face.",
    "how to sign love": "Cross both fists over chest.",
    "how to sign I love you": "Thumb, index, pinky finger up — middle and ring folded.",

    # Family
    "how to sign father": "Thumb on forehead with spread fingers.",
    "how to sign mother": "Thumb on chin with spread fingers.",
    "how to sign brother": "Make fists and tap one over the other twice.",
    "how to sign sister": "Make 'L' shape with both hands and tap at the chin.",
    "how to sign friend": "Hook index fingers together in both directions.",
    "how to sign baby": "Pretend to rock a baby in arms.",
    "how to sign uncle": "Twist 'U' hand near forehead.",
    "how to sign aunt": "Twist 'A' hand near chin.",

    # Colors
    "how to sign red": "Brush index finger down lips.",
    "how to sign blue": "Shake 'B' hand near shoulder.",
    "how to sign green": "Shake 'G' hand near shoulder.",
    "how to sign yellow": "Shake 'Y' hand near shoulder.",
    "how to sign black": "Index finger draws across forehead.",
    "how to sign white": "Pull flat hand away from chest like grabbing shirt.",
    "how to sign pink": "Middle finger brushes down chin.",
    "how to sign orange": "Squeeze fist in front of mouth (like juicing).",

    # Numbers
    "how to sign zero": "Make an 'O' with fingers.",
    "how to sign one": "Index finger up.",
    "how to sign two": "Index and middle finger up.",
    "how to sign three": "Thumb, index, middle finger up.",
    "how to sign four": "All except thumb up.",
    "how to sign five": "All fingers spread.",
    "how to sign six": "Thumb touches pinky.",
    "how to sign seven": "Thumb touches ring finger.",
    "how to sign eight": "Thumb touches middle finger.",
    "how to sign nine": "Thumb touches index finger.",

    # Functional / Common
    "how to sign yes": "Fist nods like a head.",
    "how to sign no": "Index and middle finger tap thumb (like saying 'no').",
    "how to sign help": "Thumbs up on palm, raise both hands.",
    "how to sign school": "Clap hands horizontally once.",
    "how to sign book": "Palms together, then open like a book.",
    "how to sign work": "Make fists and tap wrist together.",
    "how to sign eat": "Fingers tap mouth.",
    "how to sign drink": "Thumb to mouth as if holding a cup.",
    "how to sign water": "Tap 'W' shape at chin.",
    "how to sign bathroom": "Shake 'T' sign (thumb between fingers)."
}

    for key in q_responses:
        if key in prompt:
            return q_responses[key]
    return "🤖 Amazon Q: I'm still learning that sign! Try asking about greetings, family members, or colors."

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📥  Text/Voice ➡️ Sign Video",
    "🤖  Ask AI Tutor (with Voice)",
    "🖐️  Sign ➡️ Text + Voice"
])

# --- TAB 1 ---
with tab1:
    st.markdown("<h2 style='color:#004080;'>📥 Translate Text or Voice to Sign Language</h2>", unsafe_allow_html=True)

    input_type = st.radio("Choose Input Type", ["Text", "Microphone"])
    user_input = ""

    if input_type == "Text":
        user_input = st.text_input("Type your sentence:")

    elif input_type == "Microphone":
        duration = st.slider("🎤 Recording Duration (sec)", 2, 10, 5)
        if st.button("🔴 Start Recording"):
            audio, fs = record_audio(duration)
            save_wav("input_audio.wav", audio, fs)
            result = transcribe_audio("input_audio.wav")
            st.session_state["captured_speech"] = result
        st.text_input("🗣️ Recognized Speech", value=st.session_state.get("captured_speech", ""), key="speech_display")
        user_input = st.session_state.get("captured_speech", "")

    if st.button("▶️ Translate") and user_input:
        keywords = get_keywords(user_input)
        video_path = f"videos/{keywords[0]}.mp4"
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error("❌ No video found for this keyword.")

# --- TAB 2 ---
with tab2:
    st.markdown("<h2 style='color:#004080;'>🤖 Ask the AI Tutor</h2>", unsafe_allow_html=True)

    ask_type = st.radio("Input for AI Tutor", ["Text", "Microphone"], key="ask_input_type")
    if "ai_query" not in st.session_state:
        st.session_state["ai_query"] = ""

    if ask_type == "Text":
        st.session_state["ai_query"] = st.text_input("Ask something like 'How to sign father'", key="ai_query_text")

    elif ask_type == "Microphone":
        duration = st.slider("🎧 Ask Duration", 2, 10, 5)
        if st.button("🎙️ Record Question"):
            audio, fs = record_audio(duration)
            save_wav("ask_audio.wav", audio, fs)
            query = transcribe_audio("ask_audio.wav")
            st.session_state["ai_query"] = query
        st.text_input("🗣️ Recognized AI Query", value=st.session_state["ai_query"], key="ai_query_display")

    if st.button("🔍 Ask AI") and st.session_state["ai_query"]:
        answer = ask_amazon_q(st.session_state["ai_query"])
        st.success(answer)
        threading.Thread(target=speak_text, args=(answer,)).start()

# --- TAB 3 ---
with tab3:
    st.markdown("<h2 style='color:#004080;'>🖐️ Recognize Sign Gestures via Webcam</h2>", unsafe_allow_html=True)
    st.markdown("This will open a webcam window. Press 'q' to quit.")

    if st.button("📸 Start Gesture Recognition"):
        try:
            model = tf.keras.models.load_model("sign_model.h5")
            labels = np.load("label_classes.npy")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
            st.stop()

        phrase = ""

        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])
                        if len(coords) == 63:
                            prediction = model.predict(np.array([coords]))[0]
                            label = labels[np.argmax(prediction)]
                            confidence = np.max(prediction)
                            if confidence > 0.6:
                                phrase = label.replace("_", " ")
                                cv2.putText(frame, phrase, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                threading.Thread(target=speak_text, args=(phrase,)).start()

                cv2.imshow("Sign Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        if phrase:
            st.success(f"✅ Last recognized phrase: {phrase}")
        else:
            st.warning("⚠️ No phrase confidently recognized. Try again with better lighting or gesture.")
