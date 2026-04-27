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

VOICE_EN = "en-IN-AnanyaNeural"
VOICE_TA = "ta-IN-PallaviNeural"

# =========================
# INIT AUDIO
# =========================
pygame.mixer.init(frequency=24000, size=-16, channels=1)

# =========================
# MODELS
# =========================
print("🚀 Loading Ear (Whisper)...")
stt = WhisperModel("turbo", device="cuda", compute_type="float16")

print("🧠 Loading Brain (Gemma)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=40,
    n_threads=8,
    verbose=False
)

# =========================
# GLOBAL CONTROL & ASYNC

# =========================
is_speaking = threading.Event()
async_loop = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_loop, args=(async_loop,), daemon=True).start()

text_queue = queue.Queue()
audio_queue = queue.Queue()

# =========================
# UTILITIES
# =========================
def detect_language(text):
    return "ta" if re.search(r'[\u0b80-\u0bff]', text) else "en"

def normalize_tamil_input(text, lang):
    t = text.lower()
    tamil_words = ["enna", "sollu", "kadhai", "oru", "thamizh", "tamil", "epadi", "venum"]
    if lang != "ta" and any(w in t for w in tamil_words):
        return f"User said: {text}. Translate this intent to proper Tamil and respond fully in Tamil."
    return text

def clean_text(text):
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("*", "").replace("#", "")
    # Support both English and Tamil characters
    text = re.sub(r"[^\w\s\u0b80-\u0bff.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# TTS PIPELINE (THE MOUTH)
# =========================
async def generate_audio(text):
    voice = VOICE_TA if re.search(r'[\u0b80-\u0bff]', text) else VOICE_EN
    rate = "+10%" if voice == VOICE_TA else "+30%"

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return io.BytesIO(audio_bytes) if audio_bytes else None
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
        return None

def tts_worker():
    while True:
        text = text_queue.get()
        if text:
            future = asyncio.run_coroutine_threadsafe(generate_audio(text), async_loop)
            try:
                audio_stream = future.result(timeout=10)
                if audio_stream:
                    audio_queue.put(audio_stream)
            except Exception as e:
                print(f"⚠️ TTS Worker Timeout: {e}")
        text_queue.task_done()

def audio_player():
    while True:
        audio_stream = audio_queue.get()
        is_speaking.set()
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)
        is_speaking.clear()
        audio_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=audio_player, daemon=True).start()

# =========================
# RECORDING (THE EAR)
# =========================
def record_until_silent():
    # Wait for AI to finish talking before opening the mic
    while is_speaking.is_set() or not text_queue.empty() or not audio_queue.empty():
        time.sleep(0.1)

    print("\n👂 Listening...", flush=True)
    audio_data = []
    silent_chunks = 0
    max_silent = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    started = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            if np.linalg.norm(chunk) / np.sqrt(CHUNK_SIZE) > SILENCE_THRESHOLD:
                started = True
                silent_chunks = 0
            elif started:
                silent_chunks += 1
            
            if started: audio_data.append(chunk)
            if started and silent_chunks > max_silent: break
                
    return np.concatenate(audio_data).flatten().astype(np.float32)

# =========================
# STREAMING INFERENCE
# =========================
def run_streaming_response(user_text):
    lang = detect_language(user_text)
    user_text = normalize_tamil_input(user_text, lang)
    lang_rule = "Respond only in Tamil." if lang == "ta" else "Respond only in English."

    prompt = (f"<|im_start|>system\nYou are a sensory voice assistant for the visually impaired. "
              f"Be descriptive and clear. {lang_rule}<|im_end|>\n"
              f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")

    stream = llm(prompt, stream=True, max_tokens=400, temperature=0.7, stop=["<|im_end|>"])
    
    buffer = ""
    print("Assistant: ", end="", flush=True)
    for chunk in stream:
        token = chunk["choices"][0]["text"]
        print(token, end="", flush=True)
        buffer += token
        
        # Stream sentence by sentence for zero-latency
        if any(p in buffer for p in [". ", "! ", "? ", "\n"]):
            clean_s = clean_text(buffer)
            if len(clean_s) > 20:
                text_queue.put(clean_s)
                buffer = ""
    
    if buffer.strip():
        text_queue.put(clean_text(buffer))

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    text_queue.put("Hello. I am ready to assist you. நான் தயாராக இருக்கிறேன்.")

    while True:
        try:
            audio = record_until_silent()
            segments, _ = stt.transcribe(audio, beam_size=1, vad_filter=True)
            user_input = " ".join([s.text for s in segments]).strip()

            if not user_input or len(user_input) < 2:
                continue

            print(f"\nUser: {user_input}")

            if any(w in user_input.lower() for w in ["exit", "stop", "வெளியேறு"]):
                text_queue.put("Goodbye. Shutting down.")
                while is_speaking.is_set(): time.sleep(0.1)
                break

            run_streaming_response(user_input)

        except Exception as e:
            print(f"❌ Error: {e}")