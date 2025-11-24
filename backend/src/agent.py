import logging
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

LOG_FILE = "wellness_log.json"


# -------- JSON PERSISTENCE -------- #
def read_last_checkin():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
            if "sessions" in data and len(data["sessions"]) > 0:
                return data["sessions"][-1]
    except:
        return None
    return None


def write_checkin(entry):
    try:
        if not os.path.exists(LOG_FILE):
            data = {"sessions": []}
        else:
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
    except:
        data = {"sessions": []}

    data["sessions"].append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


# -------- WELLNESS AGENT -------- #
class WellnessAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a supportive daily Health & Wellness Voice Companion. "
                "Do a short check-in one question at a time. "
                "Do NOT provide diagnostic or medical advice.\n\n"
                "Conversation flow:\n"
                "1) Ask how the user is feeling emotionally\n"
                "2) Ask about their energy today\n"
                "3) Ask for 1–3 goals for today\n"
                "4) Offer simple realistic encouragement\n"
                "5) Recap mood + energy + goals and confirm, then stop\n"
                "Tone: warm, concise, encouraging."
            )
        )
        self.stage = "mood"
        self.prev = read_last_checkin()
        self.mood = None
        self.energy = None
        self.goals = None

    async def on_start(self, session: AgentSession):
        if self.prev:
            await session.say(
                f"Welcome back! Last time you mentioned feeling '{self.prev['mood']}' "
                f"with '{self.prev['energy']}' energy. How are you feeling today emotionally?",
                allow_interruptions=True,
            )
        else:
            await session.say(
                "Hi, good to see you again! How are you feeling today emotionally?",
                allow_interruptions=True,
            )

    async def on_response(self, response, session: AgentSession):
        msg = response.text.strip()

        if self.stage == "mood":
            self.mood = msg
            self.stage = "energy"
            return await session.say(
                "Thank you. How is your energy today — low, medium, or high?",
                allow_interruptions=True,
            )

        if self.stage == "energy":
            self.energy = msg
            self.stage = "goals"
            return await session.say(
                "Got it. What are 1–3 goals you want to focus on today?",
                allow_interruptions=True,
            )

        if self.stage == "goals":
            self.goals = [g.strip() for g in msg.split(",")] if msg else []
            self.stage = "recap"
            return await session.say(
                "Those sound good. Tip: small steps and short breaks can help. Ready for your recap?",
                allow_interruptions=True,
            )

        if self.stage == "recap":
            summary = (
                f"You're feeling '{self.mood}' with '{self.energy}' energy, "
                f"and your goals today are: {', '.join(self.goals)}."
            )

            await session.say(summary + " Does that sound right?", allow_interruptions=True)

            write_checkin(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mood": self.mood,
                    "energy": self.energy,
                    "goals": self.goals,
                    "summary": summary,
                }
            )

            self.stage = "done"
            return

        if self.stage == "done":
            await session.say("Perfect. I'll check in with you again tomorrow 💛", allow_interruptions=True)


# -------- WORKER SETUP -------- #
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(),   # 🔥 auto-selects a supported model (no 404 issues)
        tts=murf.TTS(
            voice="Anusha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=WellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.current_agent.on_start(session)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
