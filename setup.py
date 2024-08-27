from setuptools import find_packages, setup

setup(
    name="multilingual assistant",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["SpeechRecognition","gTTS","google-generativeai","python-dotenv","streamlit","requests","googlemaps","google-search-results","playsound"]
)