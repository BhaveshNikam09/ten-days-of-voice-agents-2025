import logging
import json

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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly barista at StarBrew Coffee. "
                "Your job is to take coffee orders step-by-step. "
                "You must maintain an internal order state: drinkType, size, milk, extras, and name. "
                "Ask only ONE question at a time until all fields are completed. "
                "Be warm, enthusiastic, and customer-friendly."
            )
        )

        self.order = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None,
        }

    async def on_response(self, response, session: AgentSession):
        user_msg = response.text.lower().strip()

        # Step-by-step fill order fields
        if self.order["drinkType"] is None:
            self.order["drinkType"] = user_msg
            return await session.send_response("Great choice! What size do you prefer — Small, Medium, or Large?")

        if self.order["size"] is None:
            self.order["size"] = user_msg
            return await session.send_response("Nice! What type of milk would you like — whole, oat, soy, or almond?")

        if self.order["milk"] is None:
            self.order["milk"] = user_msg
            return await session.send_response("Got it! Would you like any extras like whipped cream, caramel, vanilla? If none, say 'no extras'.")

        if self.order["extras"] == []:
            if "no" in user_msg:
                self.order["extras"] = []
            else:
                self.order["extras"] = [e.strip() for e in user_msg.split(",")]
            return await session.send_response("Awesome! May I have your name for the cup?")

        if self.order["name"] is None:
            self.order["name"] = user_msg.title()

            confirmation = (
                f"Thank you {self.order['name']}! "
                f"Your {self.order['size']} {self.order['drinkType']} "
                f"with {self.order['milk']} milk"
                + (f" and extras {', '.join(self.order['extras'])}" if self.order['extras'] else "")
                + " is on the way! ☕"
            )

            await session.send_response(confirmation)
            self.save_order()
            return

    def save_order(self):
        """Appends order to JSON file locally."""
        with open("orders.json", "a") as f:
            json.dump(self.order, f)
            f.write("\n")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
