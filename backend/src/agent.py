import json
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, metrics, tokenize
from livekit.agents import RoomInputOptions, MetricsCollectedEvent
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import uuid
import json

load_dotenv(".env.local")

CONTENT_FILE = "shared-data/razorpay_faq.json"
LEADS_FILE = "shared-data/collected_leads.json"


def load_content():
    with open(CONTENT_FILE, "r") as f:
        return json.load(f)


def save_lead(data):
    try:
        leads = []
        try:
            with open(LEADS_FILE, "r") as f:
                leads = json.load(f)
        except:
            pass
        leads.append(data)
        with open(LEADS_FILE, "w") as f:
            json.dump(leads, f, indent=2)
    except Exception as e:
        print("Lead save error:", e)


class SDR(Agent):
    def __init__(self):
        super().__init__(instructions="You are a polite Sales Development Representative for Razorpay.")
        self.content = load_content()
        self.lead = {
            "id": str(uuid.uuid4()),
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None
        }
        self.topic_asked = False

    async def on_start(self, session: AgentSession):
        await session.say(
            "👋 Hey! Welcome to Razorpay. What brought you here today?", allow_interruptions=True
        )

    def search_faq(self, msg: str):
        msg = msg.lower()
        for f in self.content["faq"]:
            if any(keyword in msg for keyword in f["question"].lower().split()):
                return f["answer"]
        return None

    async def collect_field(self, session, msg, field, prompt):
        if self.lead[field] is None:
            self.lead[field] = msg
            await session.say(prompt)
            return True
        return False

    async def on_response(self, response, session: AgentSession):
        msg = response.text.strip()

        # detect goodbye → end call
        if any(x in msg.lower() for x in ["thanks", "bye", "that’s all", "done", "i'm done"]):
            await session.say(
                f"Thanks {self.lead['name']}! Here's a quick summary: "
                f"You are a {self.lead['role']} at {self.lead['company']}, "
                f"interested in Razorpay for {self.lead['use_case']}. "
                f"Timeline: {self.lead['timeline']}. I'll share more over email soon."
            )
            save_lead(self.lead)
            return

        # try answering FAQ
        found = self.search_faq(msg)
        if found:
            await session.say(found)

        # Lead capture flow
        if await self.collect_field(session, msg, "name", "Nice! Which company are you from?"):
            return
        if await self.collect_field(session, msg, "company", "Cool — what’s your email so we stay in touch?"):
            return
        if await self.collect_field(session, msg, "email", "Your role in the company?"):
            return
        if await self.collect_field(session, msg, "role", "What's your use case for Razorpay?"):
            return
        if await self.collect_field(session, msg, "use_case", "How big is your team?"):
            return
        if await self.collect_field(session, msg, "team_size", "When are you planning to start? (now / soon / later)"):
            return
        if await self.collect_field(session, msg, "timeline", "Great! Feel free to ask anything else."):
            return


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
        agent=SDR(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await session.current_agent.on_start(session)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
