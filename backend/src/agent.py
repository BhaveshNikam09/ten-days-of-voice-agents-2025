import os
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents import RoomInputOptions
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


# ---------- GAME MASTER AGENT ----------
class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a Game Master running a fantasy Dungeons-and-Dragons style adventure. "
                "You speak dramatically, describe scenes vividly, and guide the player through the story. "
                "You always remember past decisions, events, characters, and items mentioned earlier in the story. "
                "You NEVER break character. "
                "You end every message with a prompt for action: 'What do you do?' "
                "Keep responses short enough for voice, but immersive."
            )
        )
        self.started = False

    async def on_start(self, session: AgentSession):
        await session.say(
            "Welcome, traveler. Your adventure begins in the ancient kingdom of Eldoria. "
            "Before we start... what is your hero's name?",
            allow_interruptions=True,
        )

    async def on_response(self, response, session: AgentSession):
        text = (response.text or "").strip()
        if not text:
            return

        # Step 1: First message from player = hero name
        if not self.started:
            self.hero_name = text.title()
            self.started = True

            await session.say(
                f"{self.hero_name}... a legendary name. "
                "Your journey begins deep inside the Whispering Forest. "
                "The moon glows between the leaves, and somewhere in the darkness, a wolf howls. "
                "A narrow path lies ahead, and an abandoned wooden cabin stands to your right. "
                "You sense danger... and opportunity. "
                "What do you do?",
                allow_interruptions=True,
            )
            return

        # Normal story continuation — conversation history guides the model
        await session.say("Understood. Let me continue the story...", allow_interruptions=True)

        # Final output should be story continuation + question
        await session.say(
            f"{response.text}",  # player's decision becomes part of context
            allow_interruptions=True,
        )
        await session.say("What do you do?", allow_interruptions=True)


# ---------- LIVEKIT ENTRYPOINT ----------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(),
        tts=murf.TTS(voice="Matthew", style="Conversation"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await session.current_agent.on_start(session)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
