import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentServer, JobContext, AgentSession, Agent, room_io
from livekit.plugins import google, noise_cancellation, silero

load_dotenv(".env")


# =========================
# ASSISTANT
# =========================
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Jarvis. "
                "Be brief, professional, and address the user as Sir. "
                "Speak naturally and start responding immediately."
            )
        )


# =========================
# SERVER
# =========================
server = AgentServer()


# =========================
# SESSION
# =========================
@server.rtc_session(agent_name="jarvis-agent")
async def my_agent(ctx: JobContext):

    # 🎤 Fast voice activity detection
    vad_plugin = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.3,
    )

    # 🧠 Gemini Realtime (STREAMING BUILT-IN)
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="models/gemini-2.5-flash-native-audio-latest",
            voice="Kore",
            temperature=0.6,
            instructions=(
                "You are Jarvis. "
                "Keep responses short and clear. "
                "Start speaking immediately while thinking."
            ),
        ),
        vad=vad_plugin,
    )

    # 🔌 Connect to LiveKit room
    await ctx.connect()

    # 🔊 Start session
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    # ✅ STREAMED GREETING (NO TTS ERROR)
    await session.generate_reply(
        instructions="Say: System online. How can I help, Sir?"
    )


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    agents.cli.run_app(server)