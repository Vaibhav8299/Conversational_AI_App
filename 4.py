import streamlit as st
import os
import tempfile
import requests
import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pyaudio
import wave

# Hardcoded API keys (not recommended for production use)
SERPAPI_API_KEY = "051b76a9667a340722959b664fee5fc62927e4cd3d58e2cb045c29ac08d50ba0"
GOOGLE_API_KEY = "AIzaSyDvSU3rF_Wb_IF---YkdUh_CwQbDyl9zM4"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Gemini Pro 1.5 model using LangChain's ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Define a tool using LangChain's Tool class
def fetch_serp_data(query: str):
    api_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    response = requests.get(api_url, params=params)
    return response.json()

serp_tool = Tool(
    name="FetchSerpData",
    func=fetch_serp_data,
    description="Fetch real-time search data using SerpAPI"
)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "data"], 
    template="Analyze the following search data for the query '{query}' and generate insights: {data}"
)

# Create an LLM chain with the model and the prompt template
chain = LLMChain(llm=llm, prompt=prompt_template)

# Initialize an agent
tools = [serp_tool]
agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def load_api_key():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDvSU3rF_Wb_IF---YkdUh_CwQbDyl9zM4")
    if not GOOGLE_API_KEY:
        logging.error("GOOGLE_API_KEY not found in environment variables.")
        exit()
    genai.configure(api_key=GOOGLE_API_KEY)
    return GOOGLE_API_KEY

def voice_input():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Listening... (Speak now or wait for 5 seconds to type manually)")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        text = r.recognize_google(audio)
        logging.info(f"You said: {text}")
        return text, True  # Return True to indicate voice input was used
    except sr.WaitTimeoutError:
        logging.warning("No speech detected. You can type your input instead.")
        return None, False
    except sr.UnknownValueError:
        logging.error("Could not understand the audio")
        return None, False
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service: {e}")
        return None, False

def text_to_speech(text):
    try:
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            tts = gTTS(text=text, lang="en")
            temp_file_path = temp_file.name
            tts.save(temp_file_path)
            logging.info("AI is speaking...")

        # Play audio using PyAudio
        chunk = 1024
        wf = wave.open(temp_file_path, 'rb')
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Clean up the temporary file
        os.remove(temp_file_path)
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")
        st.error(f"Failed to convert text to speech: {e}")

def llm_model_object(user_text):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(user_text)
        result = response.text
        return result
    except Exception as e:
        logging.error(f"Failed to generate content: {e}")
        return None

def process_query(query):
    logging.info(f"Processing query: {query}")

    try:
        # Step 1: Try to get response from the LLM model
        logging.info("Attempting to get a response from the LLM model...")
        llm_response = llm_model_object(query)

        if llm_response and not needs_real_time_data(llm_response):
            logging.info("Response from LLM model obtained.")
            return llm_response

        # If LLM response does not have real-time data, fetch data from SERP API
        logging.info("Fetching real-time data from SERP API...")
        serp_data = fetch_serp_data(query)

        if serp_data:
            logging.info("Data fetched successfully. Processing data with AI agent...")
            # Process data with the agent
            result = agent.run(input={"query": query, "data": serp_data})
            return result
        else:
            logging.warning("No data was retrieved from SerpAPI.")
            return "No data was retrieved from SerpAPI."
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"

def needs_real_time_data(response):
    # Logic to determine if the response needs real-time data
    # This is a placeholder; adjust based on your specific needs
    return "I don't have the current data" in response or "real-time" in response

def main():
    api_key = load_api_key()
    logging.info(f"API Key loaded: {api_key[:5]}...{api_key[-5:]}")

    st.title("Conversational AI App")
    st.write("Ask me anything...")

    user_input = st.text_input("Type your question:")
    if st.button("Speak"):
        user_input, _ = voice_input()

    if user_input:
        response = process_query(user_input)
        if response:
            st.write(f"AI Response: {response}")
            text_to_speech(response)

        st.text_input("Type your next question here:", key="next_question")

if __name__ == '__main__':
    main()