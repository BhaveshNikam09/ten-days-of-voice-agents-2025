import json
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, metrics
from livekit.agents import RoomInputOptions, MetricsCollectedEvent
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import uuid
import os

load_dotenv(".env.local")

FRAUD_DB_FILE = "shared-data/fraud_cases.json"


def _default_fraud_cases():
    """
    Creates a few sample fraud cases with fake data.
    This will be written to FRAUD_DB_FILE if it doesn't exist yet.
    """
    return [
        {
            "id": str(uuid.uuid4()),
            "userName": "John",
            "securityIdentifier": "12345",
            "cardEnding": "4242",
            "transactionAmount": 129.99,
            "transactionCurrency": "USD",
            "transactionName": "ABC Industry",
            "transactionCategory": "e-commerce",
            "transactionSource": "alibaba.com",
            "transactionLocation": "San Francisco, USA",
            "transactionTime": "2025-11-25 14:32",
            "securityQuestion": "What is your favorite color?",
            "securityAnswer": "blue",
            "status": "pending_review",
            "outcomeNote": ""
        },
        {
            "id": str(uuid.uuid4()),
            "userName": "Sara",
            "securityIdentifier": "98765",
            "cardEnding": "8812",
            "transactionAmount": 520.0,
            "transactionCurrency": "USD",
            "transactionName": "TravelNow Flights",
            "transactionCategory": "travel",
            "transactionSource": "travelnow.com",
            "transactionLocation": "New York, USA",
            "transactionTime": "2025-11-24 09:10",
            "securityQuestion": "What city were you born in?",
            "securityAnswer": "mumbai",
            "status": "pending_review",
            "outcomeNote": ""
        }
    ]


def load_fraud_cases():
    """
    Load fraud cases database from JSON file.
    If file doesn't exist, create it with default sample data.
    """
    try:
        if not os.path.exists(os.path.dirname(FRAUD_DB_FILE)):
            os.makedirs(os.path.dirname(FRAUD_DB_FILE), exist_ok=True)

        if not os.path.exists(FRAUD_DB_FILE):
            cases = _default_fraud_cases()
            with open(FRAUD_DB_FILE, "w") as f:
                json.dump(cases, f, indent=2)
            return cases

        with open(FRAUD_DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error loading fraud DB:", e)
        # fall back to in-memory defaults if something goes wrong
        return _default_fraud_cases()


def save_fraud_cases(cases):
    """
    Persist the full fraud cases list back to the JSON database.
    """
    try:
        with open(FRAUD_DB_FILE, "w") as f:
            json.dump(cases, f, indent=2)
    except Exception as e:
        print("Error saving fraud DB:", e)


class FraudAgent(Agent):
    """
    Fraud Alert Voice Agent for a fictional bank.
    - Loads a fraud case using the customer's name.
    - Verifies the customer using a basic security question.
    - Reads out the suspicious transaction.
    - Asks if the transaction is legitimate.
    - Updates the fraud case status: confirmed_safe / confirmed_fraud / verification_failed.
    """

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a calm, professional fraud detection representative for DemoBank. "
                "You are speaking to a customer about a suspicious card transaction. "
                "Use short, clear sentences. Be reassuring and helpful. "
                "NEVER ask for full card numbers, passwords, PINs, or any credentials. "
                "Only use non-sensitive information such as the user's first name, a simple security question, "
                "and masked card information that is already in the database."
            )
        )
        self.fraud_cases = load_fraud_cases()
        self.current_case = None
        self.state = "ask_name"  # ask_name -> verify -> confirm_txn -> finished
        self.verification_attempted = False

    async def on_start(self, session: AgentSession):
        """
        Called at the start of the call/session.
        """
        await session.say(
            "Hello, this is the fraud monitoring team from DemoBank. "
            "This is a demo call about a suspicious card transaction. "
            "To begin, may I know your first name?",
            allow_interruptions=True,
        )

    def _find_case_by_name(self, name: str):
        """
        Simple lookup by userName (case-insensitive).
        """
        name = name.strip().lower()
        for case in self.fraud_cases:
            if case.get("userName", "").strip().lower() == name:
                return case
        return None

    def _update_case_in_db(self):
        """
        Replace the current_case inside the fraud_cases list and save to DB.
        """
        if not self.current_case:
            return
        updated_cases = []
        for c in self.fraud_cases:
            if c.get("id") == self.current_case.get("id"):
                updated_cases.append(self.current_case)
            else:
                updated_cases.append(c)
        self.fraud_cases = updated_cases
        save_fraud_cases(self.fraud_cases)

    async def _read_transaction_and_ask(self, session: AgentSession):
        """
        Read out the suspicious transaction and ask if it was made by the customer.
        """
        c = self.current_case
        amount = c.get("transactionAmount")
        currency = c.get("transactionCurrency", "USD")
        merchant = c.get("transactionName")
        location = c.get("transactionLocation")
        time_str = c.get("transactionTime")
        card_ending = c.get("cardEnding")

        text = (
            f"Thank you for confirming your identity. "
            f"We detected a suspicious transaction on your DemoBank card ending with {card_ending}. "
            f"The transaction is for {amount} {currency} at {merchant} in {location}, "
            f"on {time_str}. "
            f"Did you make this transaction? Please answer yes or no."
        )
        await session.say(text, allow_interruptions=True)

    async def on_response(self, response, session: AgentSession):
        """
        Handle user responses and move through the call flow.
        """
        msg = response.text.strip()
        if not msg:
            return

        # Early exit if state is finished
        if self.state == "finished":
            return

        # 1) Ask for customer name and load fraud case
        if self.state == "ask_name":
            name = msg
            case = self._find_case_by_name(name)
            if not case:
                await session.say(
                    "I’m not finding any active fraud alerts under that name in this demo system. "
                    "I’ll end the call here. Thank you for your time."
                )
                self.state = "finished"
                return

            self.current_case = case
            self.state = "verify"
            question = self.current_case.get(
                "securityQuestion",
                "For security, please confirm: what is your favorite color?",
            )
            await session.say(
                f"Hi {self.current_case.get('userName')}. "
                f"Before we continue, I need to verify your identity. {question}"
            )
            return

        # 2) Verification step
        if self.state == "verify":
            self.verification_attempted = True
            expected = (self.current_case.get("securityAnswer") or "").strip().lower()
            given = msg.strip().lower()

            if expected and expected in given:
                # Verification passed
                await session.say("Perfect, thank you. Your identity is verified.")
                self.state = "confirm_txn"
                await self._read_transaction_and_ask(session)
                return
            else:
                # Verification failed → update DB and end
                self.current_case["status"] = "verification_failed"
                self.current_case[
                    "outcomeNote"
                ] = "Verification failed. Customer could not correctly answer security question in demo."
                self._update_case_in_db()

                await session.say(
                    "I’m sorry, but the details do not match our records. "
                    "For your security, I won’t be able to discuss this transaction further. "
                    "Please contact DemoBank customer support using the number on the back of your card. "
                    "This demo call will now end."
                )
                self.state = "finished"
                return

        # 3) Confirm transaction → yes/no
        if self.state == "confirm_txn":
            text = msg.lower()

            is_yes = any(
                phrase in text
                for phrase in [
                    "yes",
                    "yeah",
                    "yup",
                    "i did",
                    "it was me",
                    "this is mine",
                    "that's mine",
                    "that is mine",
                    "i made this",
                ]
            )
            is_no = any(
                phrase in text
                for phrase in [
                    "no",
                    "nope",
                    "wasn't me",
                    "was not me",
                    "i didn't",
                    "not mine",
                    "i did not",
                    "i didn't do",
                    "fraud",
                ]
            )

            if not (is_yes or is_no):
                await session.say(
                    "I’m sorry, I didn’t quite catch that. "
                    "Please clearly answer yes if you made this transaction, or no if you did not."
                )
                return

            if is_yes:
                # Mark as safe
                self.current_case["status"] = "confirmed_safe"
                self.current_case[
                    "outcomeNote"
                ] = "Customer confirmed the transaction as legitimate in demo."
                self._update_case_in_db()

                await session.say(
                    "Thank you. I have marked this transaction as safe in our system. "
                    "Your card remains active. "
                    "If you notice anything unusual in future, please contact DemoBank immediately. "
                    "This concludes our demo call. Have a great day!"
                )
                self.state = "finished"
                return

            if is_no:
                # Mark as fraud
                self.current_case["status"] = "confirmed_fraud"
                self.current_case[
                    "outcomeNote"
                ] = (
                    "Customer denied making the transaction. "
                    "Card blocked and dispute initiated in demo flow."
                )
                self._update_case_in_db()

                await session.say(
                    "Thank you for letting us know. "
                    "I have marked this transaction as fraudulent in our system. "
                    "In this demo, your card is blocked and a dispute has been initiated. "
                    "A DemoBank representative will follow up with you shortly. "
                    "Thank you for your time and stay safe."
                )
                self.state = "finished"
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
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await session.current_agent.on_start(session)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
