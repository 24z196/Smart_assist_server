import numpy as np
import sounddevice as sd
import asyncio
import edge_tts
import io
import time
import re
import pygame
import threading
import queue

from faster_whisper import WhisperModel
from llama_cpp import Llama

# =========================
# CONFIG
# =========================
MODEL_PATH = "D:/documents/gemma/gemma-4-E4B-it-IQ4_XS.gguf"

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0
CHUNK_SIZE = 1024

VOICE_EN = "en-IN-NeerjaNeural"
VOICE_TA = "ta-IN-PallaviNeural"

# =========================
# INIT AUDIO
# =========================
pygame.mixer.init(frequency=24000, size=-16, channels=1)

# =========================
# MODELS
# =========================
stt = WhisperModel("medium", device="cuda", compute_type="float16")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_gpu_layers=40,
    n_threads=8,
    verbose=False
)

# =========================
# GLOBAL CONTROL
# =========================
is_speaking = False

# =========================
# ASYNC LOOP
# =========================
async_loop = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_loop, args=(async_loop,), daemon=True).start()

# =========================
# QUEUES
# =========================
text_queue = queue.Queue()
audio_queue = queue.Queue()

# =========================
# LANGUAGE DETECTION
# =========================
def detect_language(text):
    if re.search(r'[\u0b80-\u0bff]', text):
        return "ta"
    return "en"

# =========================
# ROMAN TAMIL HANDLING
# =========================
def normalize_tamil_input(text, lang):
    t = text.lower()

    tamil_words = [
        "enna", "sollu", "kadhai", "oru",
        "thamizh", "tamil", "epadi", "venum"
    ]

    if lang != "ta" and any(w in t for w in tamil_words):
        return f"Convert this Tamil written in English letters into proper Tamil sentence and respond fully in Tamil: {text}"

    return text

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("*", "").replace("#", "")

    # preserve Tamil characters
    text = re.sub(r"[^\w\s\u0b80-\u0bff.,!?]", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# VOICE + RATE
# =========================
def get_voice_and_rate(text):
    if re.search(r'[\u0b80-\u0bff]', text):
        return VOICE_TA, "+10%"   # Tamil slower
    return VOICE_EN, "+30%"      # English faster

# =========================
# TTS
# =========================
async def generate_audio(text):
    voice, rate = get_voice_and_rate(text)

    communicate = edge_tts.Communicate(text, voice, rate=rate)

    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    return io.BytesIO(audio_bytes)

def tts_worker():
    while True:
        text = text_queue.get()
        future = asyncio.run_coroutine_threadsafe(generate_audio(text), async_loop)
        audio_stream = future.result()
        audio_queue.put(audio_stream)

threading.Thread(target=tts_worker, daemon=True).start()

# =========================
# AUDIO PLAYER
# =========================
def audio_player():
    global is_speaking
    while True:
        audio_stream = audio_queue.get()

        is_speaking = True

        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.01)

        is_speaking = False

threading.Thread(target=audio_player, daemon=True).start()

# =========================
# RECORD
# =========================
def record_until_silent():
    global is_speaking

    while is_speaking:
        time.sleep(0.05)

    print("\n🎤 Listening...")

    audio_data = []
    silent_chunks = 0
    max_silent = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    started = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK_SIZE
    ) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            volume = np.linalg.norm(chunk) / np.sqrt(CHUNK_SIZE)

            if volume > SILENCE_THRESHOLD:
                started = True
                silent_chunks = 0
            elif started:
                silent_chunks += 1

            if started:
                audio_data.append(chunk)

            if started and silent_chunks > max_silent:
                break

    return np.concatenate(audio_data).flatten().astype(np.float32)

# =========================
# GENERATE RESPONSE
# =========================
def generate_response(user_text):
    lang = detect_language(user_text)
    user_text = normalize_tamil_input(user_text, lang)

    lang_rule = (
        "Respond only in Tamil using proper Tamil words."
        if lang == "ta"
        else "Respond only in English."
    )

    prompt = f"""
<|im_start|>system
You are a voice assistant.

Speak clearly.
Do not spell letters.
Use full natural sentences.

{lang_rule}
<|im_end|>

<|im_start|>user
{user_text}
<|im_end|>

<|im_start|>assistant
"""

    output = llm(
        prompt,
        max_tokens=120,
        temperature=0.4,
        stop=["<|im_end|>"]
    )

    return clean_text(output["choices"][0]["text"])

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    text_queue.put("Hello. நான் தயாராக இருக்கிறேன். சொல்லுங்கள்.")

    while True:
        try:
            if is_speaking:
                time.sleep(0.05)
                continue

            audio = record_until_silent()

            segments, info = stt.transcribe(audio, beam_size=1, vad_filter=True)
            user_text = " ".join([s.text for s in segments]).strip()

            # improved Tamil heuristic
            if re.search(r'(thamizh|tamil|kadhai|sollu|enna|oru|epadi|venum)', user_text.lower()):
                info.language = "ta"

            if not user_text:
                continue

            print("User:", user_text)

            if "exit" in user_text.lower() or "வெளியேறு" in user_text:
                text_queue.put("Goodbye. விடைபெறுகிறேன்.")
                break

            response = generate_response(user_text)
            text_queue.put(response)

        except Exception as e:
            print("Error:", e)