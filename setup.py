from setuptools import find_packages, setup

setup(
    name="multilingual_assistant",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "requests",
        "SpeechRecognition",
        "google-generativeai",
        "gTTS",
        "langchain",
        "setuptools>=65.0.0",
        "wheel>=0.40.0",
        "langchain_google_genai",
        "playsound"
    ],
)
