import os
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import (
      groq, 
      deepgram, 
      noise_cancellation, 
      silero
)

load_dotenv(".env")
load_dotenv(".env.local")

class Assistant(Agent):
      def __init__(self) -> None:
            super().__init__(instructions=(
                        "You are Jarvis, a sophisticated and friendly AI assistant. "
                        "Your primary role is to assist users with a wide range of tasks, providing accurate and helpful information in a conversational manner. "
                        "Dont make the answer too long, keep it concise and to the point. Not too short at the same time."
                  ))

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
      # 1. Initialize VAD (Voice Activity Detection)
      vad_plugin = silero.VAD.load()

      # 2. Setup the Groq Pipeline
      session = AgentSession(
            llm=groq.LLM(model="llama-3.3-70b-versatile"), # Fast and usually free/cheap
            # To auto-detect between multiple languages:
            stt=deepgram.STT(model="nova-3", language="multi"),                                  
            tts=deepgram.TTS(model="aura-2-orion-en"),
            vad=vad_plugin,
      )
      
      # 3. Connect to the room
      await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_options=room_io.RoomOptions(
                  audio_input=room_io.AudioInputOptions(
                        noise_cancellation=lambda params: noise_cancellation.BVC(),
                  ),
            ),
      )
      
      # 4. Immediate Greeting
      await session.say("Jarvis is back online using Groq. How can I help you?")

if __name__ == "__main__":
      agents.cli.run_app(server)

