import asyncio
import json
import logging
import sys
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from collections import deque

from dotenv import load_dotenv
import httpx

load_dotenv()

import anthropic

import logging

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=getenv("ANTHROPIC_API_KEY"))

CAL_API_KEY = getenv("CAL_API_KEY")
CAL_BASE_URL = "https://api.cal.com/v2"

TOKEN = getenv("BOT_TOKEN")


from datetime import datetime, timezone, timedelta

KYIV_TZ = timezone(timedelta(hours=3))


def format_slots_for_claude(raw_response: dict) -> dict:
    if "data" not in raw_response:
        return raw_response

    formatted = {}
    for date, slots in raw_response["data"].items():
        formatted[date] = [
            {
                "start_utc": slot["start"],
                "start_kyiv": datetime.fromisoformat(
                    slot["start"].replace("Z", "+00:00")
                )
                .astimezone(KYIV_TZ)
                .strftime("%H:%M"),
            }
            for slot in slots
        ]
    return {"data": formatted}


async def get_available_slots(event_type_id, start, end):
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            f"{CAL_BASE_URL}/slots",
            headers={
                "Authorization": f"Bearer {CAL_API_KEY}",
                "cal-api-version": "2024-09-04",
            },
            params={
                "eventTypeId": event_type_id,
                "start": start,
                "end": end,
            },
        )
        return format_slots_for_claude(response.json())


async def create_booking(event_type_id, start, name, email):
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{CAL_BASE_URL}/bookings",
            headers={
                "Authorization": f"Bearer {CAL_API_KEY}",
                "cal-api-version": "2026-02-25",
            },
            json={
                "eventTypeId": event_type_id,
                "start": start,
                "attendee": {"name": name, "email": email, "timeZone": "Europe/Kyiv"},
            },
        )
        return response.json()


chat_histories = {}

system_prompt = """
You are a professional assistant for Ilya Vasylchenko, an AI Automation Consultant.

## About the Consultant
Name: Ilya Vasylchenko
Specialty: AI automation for small and medium businesses
Location: Ukraine (remote worldwide)

## Services
1. AI Automation Audit ($100) — 60 min session
   - Analyze current business processes
   - Identify automation opportunities
   - Deliver prioritized action plan

2. Custom AI Workflow Setup (negotiable) — full implementation
   - n8n automation workflows
   - Claude/GPT API integration
   - Telegram bots for business

3. Monthly AI Support ($300/mo)
   - Maintain and improve automations
   - Add new integrations
   - Priority support

## Your Job
- Answer questions about services clearly and concisely
- Qualify the client: ask about their business and main pain point
- When client wants to book — collect: name, email, preferred date/time
- Always be professional, friendly, and solution-focused

## Booking Flow
When client wants to schedule:
1. Ask for their name
2. Ask for their email
3. Ask for preferred date and time
4. Confirm details before booking
5. Use the booking tool to create appointment

## Current date
Today is {current_date}. Use this to interpret 
"tomorrow", "next week", etc. when client mentions preferred time.

## Event Type IDs  
- AI Automation Audit (60 min): ID = 5165179
- Custom Workflow Setup (30 min discovery call): ID = 4921514

## Rules
- Never invent services or prices not listed above
- If unsure — say "Let me clarify that for you"
- Keep responses short (3-5 sentences max)
- Always end with a clear next step or question
- Answer using HTML, such as <b>, <i>, <code> only.
"""

TOOLS = [
    {
        "name": "get_available_slots",
        "description": """Search for available slots. RULES:
         - eventTypeId — numeric ID from the catalog
         - start/end — ISO8601 UTC: 2026-06-15T00:00:00Z / 2026-06-15T23:59:00Z
         - Each slot has: start_kyiv (show to client) and start_utc (use for create_booking)
         - ALWAYS show client start_kyiv time
         - ALWAYS pass start_utc to create_booking, never start_kyiv""",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_type_id": {
                    "type": "integer",
                    "description": "Numeric event type ID",
                },
                "start": {
                    "type": "string",
                    "description": "Start of search range ISO8601 UTC",
                },
                "end": {
                    "type": "string",
                    "description": "End of search range ISO8601 UTC",
                },
            },
            "required": ["event_type_id", "start", "end"],
        },
    },
    {
        "name": "create_booking",
        "description": """Create a booking. Call ONLY if:
    ✅ client confirmed a specific slot
    ✅ name and email are present
    - start — copy BYTE-FOR-BYTE from get_available_slots
    - FORBIDDEN to reformat the timestamp""",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_type_id": {
                    "type": "integer",
                    "description": "Numeric event type ID",
                },
                "start": {
                    "type": "string",
                    "description": "Exact slot timestamp from get_available_slots",
                },
                "name": {"type": "string", "description": "Client full name"},
                "email": {
                    "type": "string",
                    "description": "Client email, must contain @",
                },
            },
            "required": ["event_type_id", "start", "name", "email"],
        },
    },
]

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    chat_histories[message.from_user.id] = deque(maxlen=20)  # clear history
    await message.answer(
        "<b>Hi! I'm Ilya's AI assistant.</b>\n\n"
        "I can help you:\n"
        "• Learn about AI automation services\n"
        "• Book a consultation\n\n"
        "What brings you here today?"
    )


@dp.message()
async def echo_handler(message: Message, bot: Bot) -> None:
    user_id = message.from_user.id
    formatted_system = system_prompt.format(
        current_date=datetime.now().strftime("%A, %B %d, %Y")
    )
    if user_id not in chat_histories:
        chat_histories[user_id] = deque(maxlen=20)

    chat_histories[user_id].append({"role": "user", "content": message.text})

    messages = list(chat_histories[user_id])

    await bot.send_chat_action(message.chat.id, action="typing")

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=formatted_system,
        tools=TOOLS,
        messages=messages,
    )

    # while loop
    while response.stop_reason == "tool_use":
        # find correct tool
        tool_block = next(b for b in response.content if b.type == "tool_use")

        # user tool
        if tool_block.name == "get_available_slots":
            tool_result = await get_available_slots(**tool_block.input)
        elif tool_block.name == "create_booking":
            tool_result = await create_booking(**tool_block.input)
        else:
            tool_result = {"error": f"Unknown tool: {tool_block.name}"}

        logger.info(
            f"Tool call: {tool_block.name} | Input: {tool_block.input} | User: {user_id}"
        )
        logger.info(f"Tool result: {json.dumps(tool_result)[:200]}")

        await bot.send_chat_action(message.chat.id, action="typing")

        # save messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": json.dumps(tool_result),
                    }
                ],
            }
        )

        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=formatted_system,
            tools=TOOLS,
            messages=messages,
        )

    await bot.send_chat_action(message.chat.id, action="typing")

    reply = next(b.text for b in response.content if b.type == "text")

    # save final messages history
    chat_histories[user_id] = deque(
        messages + [{"role": "assistant", "content": reply}], maxlen=20
    )

    await message.answer(reply, parse_mode=ParseMode.HTML)


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
