import os
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, cli
from livekit.plugins import (
    groq, 
    deepgram, 
    google, 
    silero
)

load_dotenv(".env")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=(
            "You are Jarvis, a sophisticated AI assistant. "
            "You are powered by Groq (thinking) and Gemini (speaking). "
            "You are fluent in Tamil and English. "
            "Respond in the same language the user speaks to you. "
            "Keep answers concise and helpful."
        ))

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
    # 1. Groq Brain
    groq_llm = groq.LLM(model="llama-3.3-70b-versatile")
    
    # 2. Deepgram Ears (No billing/JSON needed)
    deepgram_stt = deepgram.STT(model="nova-3", language="multi")

    # 3. Gemini Mouth (The CORRECT 2026 Class Name)
    # This class is specifically built for GOOGLE_API_KEY from AI Studio.
    tts = google.beta.GeminiTTS(model="gemini-2.5-flash-preview-tts")

    # 4. Connect everything in the Session
    session = AgentSession(
        llm=groq_llm,
        stt=deepgram_stt,
        tts=tts,
        vad=silero.VAD.load(),
    )

    # 5. Start and speak
    await session.start(room=ctx.room, agent=Assistant())
    
    # Greeting
    await session.say("Jarvis is online. Groq brain and Gemini voice active. Epdi irukinga?")

if __name__ == "__main__":
    cli.run_app(server)