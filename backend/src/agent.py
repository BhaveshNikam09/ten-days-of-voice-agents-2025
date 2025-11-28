import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents import RoomInputOptions, MetricsCollectedEvent
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

# ---------- FILE PATHS ----------
CATALOG_FILE = "shared-data/catalog.json"
ORDERS_DB_FILE = "shared-data/orders.json"

BRAND_NAME = "QuickFresh"  # brand name for the assistant


# ---------- CATALOG & ORDERS HELPERS ----------
def _ensure_shared_data_dir():
    dirname = os.path.dirname(CATALOG_FILE)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def load_catalog() -> List[Dict[str, Any]]:
    _ensure_shared_data_dir()
    if not os.path.exists(CATALOG_FILE):
        # Create a blank catalog if missing
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_orders() -> List[Dict[str, Any]]:
    _ensure_shared_data_dir()
    if not os.path.exists(ORDERS_DB_FILE):
        with open(ORDERS_DB_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []
    with open(ORDERS_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_orders(orders: List[Dict[str, Any]]):
    _ensure_shared_data_dir()
    with open(ORDERS_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2)


# ---------- SIMPLE RECIPE MAPPING ----------
RECIPES = {
    "peanut butter sandwich": {},
    "pasta": {},
    "simple breakfast": {},
    "salad": {},
    "cereal breakfast": {},
}

# ---------- ORDER STATUS FLOW ----------
STATUS_FLOW = [
    ("received", 0),
    ("confirmed", 2),
    ("being_prepared", 8),
    ("out_for_delivery", 20),
    ("delivered", 40),
]


def parse_int_from_text(text: str) -> Optional[int]:
    text = text.lower()
    for tok in text.split():
        if tok.isdigit():
            return int(tok)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for w, n in words.items():
        if w in text:
            return n
    return None


def update_order_status_based_on_time(order: Dict[str, Any]) -> None:
    try:
        created_dt = datetime.fromisoformat(order.get("created_at"))
        minutes = (datetime.now() - created_dt).total_seconds() / 60.0
        best_status = order.get("status", "received")
        for status, min_minutes in STATUS_FLOW:
            if minutes >= min_minutes:
                best_status = status
        order["status"] = best_status
    except:
        pass


# ---------- AGENT IMPLEMENTATION ----------
class GroceryOrderingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                f"You are a friendly food and grocery ordering assistant for {BRAND_NAME}. "
                "Help customers add items to a cart, show cart, and place orders. "
                "After checkout, allow tracking of order status using stored JSON data."
            )
        )
        self.catalog = load_catalog()
        self.orders = load_orders()
        self.customer_name: Optional[str] = None
        self.cart: Dict[str, float] = {}
        self.state: str = "ask_name"

    # --- Utility ---
    def _get_item(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.lower()
        for item in self.catalog:
            if text in item["name"].lower() or text in item["id"].lower():
                return item
        return None

    def _cart_items_detailed(self):
        items = []
        for item_id, qty in self.cart.items():
            item = next((i for i in self.catalog if i["id"] == item_id), None)
            if item:
                items.append(
                    {
                        "id": item_id,
                        "name": item["name"],
                        "price": item["price"],
                        "quantity": qty,
                        "line_total": item["price"] * qty,
                    }
                )
        return items

    def _add_order(self):
        order_id = str(uuid.uuid4())
        detailed_items = self._cart_items_detailed()
        total = sum(i["line_total"] for i in detailed_items)
        order = {
            "id": order_id,
            "customer_name": self.customer_name,
            "items": detailed_items,
            "total": total,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "received",
        }
        self.orders.append(order)
        save_orders(self.orders)
        return order

    # --- Lifecycle ---
    async def on_start(self, session: AgentSession):
        await session.say(
            f"Hi, welcome to {BRAND_NAME}! May I know your name?",
            allow_interruptions=True,
        )

    async def on_response(self, response, session: AgentSession):
        text = (response.text or "").lower().strip()
        if not text:
            return

        # Step 1: Ask for name
        if self.state == "ask_name":
            self.customer_name = text.strip().title()
            self.state = "ordering"
            await session.say(
                f"Nice to meet you, {self.customer_name}. What would you like to order first?",
                allow_interruptions=True,
            )
            return

        # Show cart
        if "cart" in text:
            if not self.cart:
                await session.say("Your cart is empty.", allow_interruptions=True)
            else:
                msg = "; ".join(f"{d['quantity']} x {d['name']}" for d in self._cart_items_detailed())
                await session.say(f"In your cart: {msg}", allow_interruptions=True)
            return

        # Place order
        if "checkout" in text or "place my order" in text:
            if not self.cart:
                await session.say("Your cart is empty.", allow_interruptions=True)
                return
            order = self._add_order()
            self.cart.clear()
            self.state = "finished"
            await session.say(
                f"Your order is placed! Order ID: {order['id']}. Status set to received.",
                allow_interruptions=True,
            )
            return

        # Tracking
        if "track" in text or "where is my order" in text:
            if not self.orders:
                await session.say("No previous order found.", allow_interruptions=True)
                return
            order = self.orders[-1]
            update_order_status_based_on_time(order)
            save_orders(self.orders)
            await session.say(
                f"Your latest order status is '{order['status']}'.",
                allow_interruptions=True,
            )
            return

        # Add items
        item = self._get_item(text)
        qty = parse_int_from_text(text) or 1
        if item:
            self.cart[item["id"]] = self.cart.get(item["id"], 0) + qty
            await session.say(
                f"Added {qty} × {item['name']} to your cart.",
                allow_interruptions=True,
            )
            return

        await session.say(
            "I'm not sure what you mean. Please say an item name from the catalog.",
            allow_interruptions=True,
        )


# ---------- LIVEKIT SETUP ----------
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
        agent=GroceryOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await session.current_agent.on_start(session)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
