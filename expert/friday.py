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
import re
import subprocess
import requests
from datetime import datetime

# Name of the model you pulled with `ollama pull <model_name>`
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"


def speak(text):
    """Converts text to speech and plays it out loud."""
    print(f"Friday: {text}")
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


# Common websites you can open just by saying their name, e.g. "open youtube"
# instead of having to say the full domain. Add more entries as you like.
WEBSITE_SHORTCUTS = {
    "youtube": "https://youtube.com",
    "google": "https://google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "reddit": "https://reddit.com",
    "amazon": "https://amazon.com",
    "netflix": "https://netflix.com",
}

# Matches "open <something>", optionally followed by "in chrome" / "on chrome"
OPEN_COMMAND_PATTERN = re.compile(r"^open\s+(.+?)(?:\s+(?:in|on)\s+chrome)?$")


def looks_like_a_domain(text):
    """Returns True for things like 'reddit.com' or 'sub.example.co' — a word
    containing at least one dot, with no spaces, and no dot at the very start/end."""
    return bool(re.match(r"^[\w-]+(\.[\w-]+)+$", text))


def get_current_time():
    """Returns the current local time as a natural-sounding phrase."""
    now = datetime.now()
    return f"It's currently {now.strftime('%I:%M %p')}."


def get_weather(city="your area"):
    """
    Fetches current weather from wttr.in (a free, no-API-key weather service).
    '%C' = condition (e.g. 'Sunny'), '%t' = temperature.
    """
    try:
        # If no specific city was mentioned, wttr.in auto-detects location from IP
        url = f"https://wttr.in/{city}?format=%C+%t" if city != "your area" else "https://wttr.in?format=%C+%t"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return f"The weather right now is {response.text.strip()}."
    except requests.exceptions.RequestException:
        return "I couldn't fetch the weather right now. Check your internet connection."


def handle_command(text):
    """
    Checks if the spoken text matches a recognized command (time, weather,
    or "open X"). If it does, executes it directly and returns a confirmation
    string. If it isn't a recognized command, returns None so the caller
    knows to fall through to the normal LLM conversation instead.
    """
    cleaned = text.lower().strip().rstrip(".,!?")

    # --- Time command ---
    if "what" in cleaned and "time" in cleaned:
        return get_current_time()

    # --- Weather command ---
    if "weather" in cleaned:
        # crude check for "weather in <city>" — grabs whatever comes after "in"
        weather_match = re.search(r"weather in (.+)", cleaned)
        city = weather_match.group(1).strip() if weather_match else "your area"
        return get_weather(city)

    # --- Open command (chrome / website / app) ---
    match = OPEN_COMMAND_PATTERN.match(cleaned)
    if not match:
        return None  # not a recognized command at all — let Ollama handle it as normal chat

    target = match.group(1).strip()

    # Case 1: just "open chrome" / "open google chrome" — launch the browser itself
    if target in ("chrome", "google chrome"):
        return run_open_command(["-a", "Google Chrome"], "Opening Chrome.")

    # Case 2: a known website shortcut, e.g. "open youtube"
    if target in WEBSITE_SHORTCUTS:
        url = WEBSITE_SHORTCUTS[target]
        return run_open_command(["-a", "Google Chrome", url], f"Opening {target} in Chrome.")

    # Case 3: something that looks like a raw domain, e.g. "open reddit.com"
    if looks_like_a_domain(target) or target.startswith("http"):
        url = target if target.startswith("http") else f"https://{target}"
        return run_open_command(["-a", "Google Chrome", url], f"Opening {url} in Chrome.")

    # Case 4: fall back to treating it as a Mac application name, e.g. "open spotify"
    return run_open_command(["-a", target], f"Opening {target}.")


def run_open_command(args, success_message):
    """
    Runs macOS's built-in `open` command with the given arguments.
    Returns a spoken confirmation on success, or an error message on failure.
    """
    try:
        result = subprocess.run(
            ["open"] + args,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return success_message
        else:
            return f"I couldn't open that. macOS said: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "That took too long to open."
    except Exception as e:
        return f"Something went wrong trying to open that: {e}"


def main():
    print("Voice assistant ready. Say 'exit' or 'quit' to stop.")
    speak("Hello! I'm ready. What would you like to ask?")

    while True:
        user_input = listen_and_transcribe()

        if user_input is None:
            # Nothing understood this round — just loop back and listen again
            continue

        # Clean up the transcribed text before checking for exit words:
        # - lowercase, so "Exit" and "exit" both match
        # - strip whitespace from the ends
        # - remove common trailing punctuation Google's STT sometimes adds
        cleaned_input = user_input.lower().strip().rstrip(".,!?")

        # Use "in" instead of exact equality so phrases like "okay stop" or
        # "please exit now" still trigger the exit, not just a bare "exit"
        if any(word in cleaned_input.split() for word in ["exit", "quit", "stop"]):
            speak("Goodbye!")
            break

        # Check if this is a recognized "open X" command first —
        # only fall through to the LLM if it isn't a command at all
        command_result = handle_command(user_input)
        if command_result is not None:
            speak(command_result)
            continue

        ai_reply = ask_ollama(user_input)
        speak(ai_reply)


if __name__ == "__main__":
    main()