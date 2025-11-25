import json
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, metrics, tokenize
from livekit.agents import RoomInputOptions, MetricsCollectedEvent
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

CONTENT_FILE = "shared-data/day4_tutor_content.json"


def load_content():
    with open(CONTENT_FILE, "r") as f:
        return json.load(f)


class TutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are an Active Recall Learning Coach. You operate in 3 modes: learn, quiz, and teach_back. "
                "The user can switch between these modes anytime simply by saying the mode name. "
                "Use short explanations and stay interactive."
            )
        )
        self.mode = None
        self.concepts = load_content()
        self.current_concept = None

    async def on_start(self, session: AgentSession):
        await session.say(
            "👋 Welcome to Sage-the-Tutor! Which mode would you like — learn, quiz, or teach-back?",
            allow_interruptions=True,
        )

    async def set_voice(self, session, voice):
        session.tts = murf.TTS(
            voice=voice,
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        )

    def pick_concept(self, msg: str):
        for c in self.concepts:
            if c["id"] in msg.lower() or c["title"].lower() in msg.lower():
                return c
        return None

    async def on_response(self, response, session: AgentSession):
        msg = response.text.strip().lower()

        # MODE CHANGE
        if "learn" in msg:
            self.mode = "learn"
            await self.set_voice(session, "Nikhil")
            return await session.say("Great — which concept do you want to learn?")

        if "quiz" in msg:
            self.mode = "quiz"
            await self.set_voice(session, "Tanushree")
            return await session.say("You got it — which concept should I quiz you on?")

        if "teach" in msg:
            self.mode = "teach_back"
            await self.set_voice(session, "Priya")
            return await session.say("Nice — which concept do you want to teach me back?")

        if self.mode is None:
            return await session.say("Please choose a mode: learn, quiz, or teach-back.")

        # Pick concept
        if self.current_concept is None:
            concept = self.pick_concept(msg)
            if concept is None:
                return await session.say(
                    "I couldn't identify the concept. Try words like variables or loops."
                )
            self.current_concept = concept

        concept = self.current_concept

        # MODE BEHAVIOR
        if self.mode == "learn":
            await session.say(f"{concept['title']}: {concept['summary']}")
            self.current_concept = None
            return await session.say("Would you like another concept or a different mode?")

        if self.mode == "quiz":
            await session.say(concept["sample_question"])
            self.current_concept = None
            return await session.say("I’ll wait for your answer — or you can switch mode anytime.")

        if self.mode == "teach_back":
            await session.say(f"Teach me: {concept['sample_question']}")
            self.current_concept = None
            return await session.say("I’ll listen — and you can switch modes anytime.")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(),
        tts=murf.TTS(voice="Matthew", style="Conversation"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await session.current_agent.on_start(session)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
