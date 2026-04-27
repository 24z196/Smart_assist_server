import os
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext, AgentSession, Agent, room_io
from livekit.plugins import (
    google,
    noise_cancellation,
    silero, 
)

load_dotenv(".env")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Jarvis. Be brief, professional, and address the user as Sir."
        )

server = AgentServer()

@server.rtc_session(agent_name="jarvis-agent")
async def my_agent(ctx: JobContext):
    
    vad_plugin = silero.VAD.load(
        min_speech_duration=0.1,  
        min_silence_duration=0.3,
    )

    # 1. FIXED MODEL ID BASED ON YOUR SUCCESSFUL CHECK
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="models/gemini-2.5-flash-native-audio-latest", 
            voice="Charon",
            temperature=0.7,
            instructions="You are Jarvis. Keep responses very short.",
        ),
        vad=vad_plugin, 
    )

    await ctx.connect()

    # 2. FIXED NOISE CANCELLATION
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    # 3. INITIAL GREETING
    await session.generate_reply(
        instructions="Greet the user as Jarvis. Say: 'System online. How can I help, Sir?'"
    )

if __name__ == "__main__":
    agents.cli.run_app(server)