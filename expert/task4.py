import speech_recognition as sr
from gtts import gTTS
import os
import requests

OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"


def speak(text):
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
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
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
            continue

        if user_input.lower().strip() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break

        ai_reply = ask_ollama(user_input)
        speak(ai_reply)


if __name__ == "__main__":
    main()