import numpy as np
import sounddevice as sd
import time
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.2   # seconds of silence to stop

# =========================
# LOAD MODEL
# =========================
print("Loading Whisper...")
model = WhisperModel(
    "turbo",           # ⚡ fast + multilingual
    device="cpu",
    compute_type="int8"
)

print("✅ Whisper Ready\n")

# =========================
# SMART RECORDING (VAD-like)
# =========================
def record_audio():
    print("🎤 Listening...")

    audio_buffer = []
    silent_time = 0
    speaking_started = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_time, speaking_started

        volume = np.linalg.norm(indata) / np.sqrt(len(indata))

        if volume > SILENCE_THRESHOLD:
            speaking_started = True
            silent_time = 0
        elif speaking_started:
            silent_time += frames / SAMPLE_RATE

        audio_buffer.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while True:
            if speaking_started and silent_time > SILENCE_DURATION:
                break

    audio = np.concatenate(audio_buffer, axis=0).flatten()

    return audio.astype(np.float32)

# =========================
# AUDIO CLEANING
# =========================
def clean_audio(audio):
    # normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # remove DC offset
    audio = audio - np.mean(audio)

    # noise gate
    threshold = 0.015
    audio[np.abs(audio) < threshold] = 0

    # smoothing
    audio = np.convolve(audio, np.ones(3)/3, mode='same')

    return audio.astype(np.float32)

# =========================
# TRANSCRIPTION
# =========================
def transcribe(audio):
    audio = clean_audio(audio)

    start_time = time.time()

    segments, info = model.transcribe(
        audio,
        beam_size=1,
        vad_filter=True,
        language=None   # 🔥 auto detect (English + Tamil)
    )

    text = " ".join([seg.text for seg in segments]).strip()

    end_time = time.time()

    return text, info.language, (end_time - start_time)

# =========================
# MAIN LOOP
# =========================
while True:
    audio = record_audio()

    text, lang, latency = transcribe(audio)

    if not text:
        print("⚠️ No speech detected\n")
        continue

    print("\n📝 Transcribed:", text)
    print("🌐 Language:", lang)
    print(f"⏱️ Processing time: {latency:.2f} sec\n")

    if "exit" in text.lower():
        print("👋 Exiting...")
        break