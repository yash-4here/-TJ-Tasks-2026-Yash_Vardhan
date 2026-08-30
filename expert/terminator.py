"""
Expert Task: Conversational Voice Pipeline (Ear -> Brain -> Mouth)
--------------------------------------------------------------------
Records your voice, transcribes it, sends it to a local Ollama model,
and speaks the reply back out loud, in a continuous loop.

SETUP REQUIRED BEFORE RUNNING:
1. Install Ollama: https://ollama.com/download
2. Pull a model in your terminal, e.g.:  ollama pull llama3.2
3. Make sure Ollama is running (it usually starts automatically after install,
   or run `ollama serve` in a terminal).
4. pip install speechrecognition gtts requests pyaudio
   (On Mac, if pyaudio fails to install: brew install portaudio, then retry pip install)
"""

import speech_recognition as sr
from gtts import gTTS
import os
import requests

# Name of the model you pulled with `ollama pull <model_name>`
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"


def speak(text):
    """Converts text to speech and plays it out loud."""
    print(f"AI: {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")

    if os.name == "nt":
        os.system("start response.mp3")
    elif os.uname().sysname == "Darwin":
        os.system("afplay response.mp3")
    else:
        os.system("mpg321 response.mp3")


def listen_and_transcribe():
    """Listens to the microphone and converts speech to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening... (speak now)")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
        return None
    except sr.RequestError:
        print("Speech recognition service error. Check your internet connection.")
        return None


def ask_ollama(prompt):
    """Sends the transcribed text to a local Ollama model and returns its reply."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False  # get the full reply at once instead of a stream of chunks
            },
            timeout=30
        )
        response.raise_for_status()  # raises an error if the request failed
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "I can't reach Ollama right now. Is it running?"
    except requests.exceptions.Timeout:
        return "That took too long to respond. Please try again."
    except Exception as e:
        return f"Something went wrong talking to the model: {e}"


def main():
    print("Voice assistant ready. Say 'exit' or 'quit' to stop.")
    speak("Hello! I'm ready. What would you like to ask?")

    while True:
        user_input = listen_and_transcribe()

        if user_input is None:
            # Nothing understood this round — just loop back and listen again
            continue

        if user_input.lower().strip() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break

        ai_reply = ask_ollama(user_input)
        speak(ai_reply)


if __name__ == "__main__":
    main()