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
        "playsound",
        "langchain"
    ],
)
