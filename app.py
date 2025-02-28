import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import os
from io import BytesIO

# Initialize AI Model
groq_api_key = st.secrets["GROQ_API_KEY"] # Replace with your actual API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-specdec", temperature=0.7)

# Function to recognize speech input
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Error: Speech Recognition service is unavailable."

# Function to generate AI response
def get_ai_response(user_input):
    response = llm.invoke([{"role": "user", "content": user_input}])
    return response.content

# Function to translate text
def translate_text(text, target_lang="en"):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Function to convert text to speech using gTTS
def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.save("output.mp3")
    with open("output.mp3", "rb") as f:
        audio_bytes.write(f.read())
    audio_bytes.seek(0)
    return audio_bytes

# Streamlit UI
st.title("🗣 AI Chatbot with Voice & Multilingual Support")
st.write("Type or speak your question below.")

# User selects language
lang_code = st.selectbox("Choose your language:", ["en", "pt", "fr", "es", "de"])
user_input = st.text_input("Type your message (or use voice input):")

# Voice Input Button
if st.button("🎙 Speak"):
    user_input = speech_to_text()
    st.write("Recognized Text:", user_input)

if st.button("📝 Send"):
    if user_input:
        # Translate input to English before processing
        translated_input = translate_text(user_input, "en")

        # Get AI response
        response = get_ai_response(translated_input)

        # Translate back to the selected language
        final_response = translate_text(response, lang_code)

        st.write("🤖 Chatbot:", final_response)

        # Convert response to speech
        audio_file = text_to_speech(final_response, lang_code)
        audio_data = base64.b64encode(audio_file.read()).decode()

        # Play audio in Streamlit
        st.audio(audio_file, format="audio/mp3")
        st.markdown(f'<audio controls><source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3"></audio>', unsafe_allow_html=True)
