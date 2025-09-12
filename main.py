from typing import Optional

from agents import Agent, Runner, RunConfig, enable_verbose_stdout_logging, ModelSettings
from agents.mcp import MCPServerSse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from openai import max_retries
from openai.types import Reasoning

from matrix_bot import MatrixChatBot

# Environment variables - read all at once
MATRIX_HOMESERVER_URL = os.getenv("MATRIX_HOMESERVER_URL")
MATRIX_USERNAME = os.getenv("MATRIX_USERNAME")
MATRIX_PASSWORD = os.getenv("MATRIX_PASSWORD")
MATRIX_ROOM_ID = os.getenv("MATRIX_ROOM_ID")
SYSTEM_USERNAME = os.getenv("SYSTEM_USERNAME")  # Username to identify as "system" in conversation context
MCP_SERVER_URL_MEMORY = os.getenv("MCP_SERVER_URL_MEMORY", "http://localhost:8300/sse")
MCP_SERVER_URL_MISC = os.getenv("MCP_SERVER_URL_MISC", "http://localhost:8100/sse")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

matrix_bot : Optional[MatrixChatBot] = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage the lifespan of the FastAPI app and Matrix bot"""
    global matrix_bot

    # Startup: Start Matrix bot if environment variables are set
    if all([MATRIX_HOMESERVER_URL, MATRIX_USERNAME, MATRIX_PASSWORD, MATRIX_ROOM_ID]):
        from matrix_bot import MatrixChatBot
        matrix_bot = MatrixChatBot(
            homeserver=MATRIX_HOMESERVER_URL,
            user_id=MATRIX_USERNAME,
            password=MATRIX_PASSWORD,
            room_id=MATRIX_ROOM_ID,
            mcp_memory_url=MCP_SERVER_URL_MEMORY,
            system_username=SYSTEM_USERNAME
        )
        # Start Matrix bot in background task
        asyncio.create_task(matrix_bot.start())
        print("Matrix bot started")
    else:
        print("Matrix bot not started - missing environment variables")

    yield

    # Shutdown: Stop Matrix bot
    if matrix_bot:
        await matrix_bot.stop()
        print("Matrix bot stopped")


app = FastAPI(title="HA AI Tasker", version="0.1.0", lifespan=lifespan)

def get_recent_conversation_context() -> str:
    """Get recent conversation history from Matrix bot for context"""
    global matrix_bot
    if not matrix_bot:
        return "No recent conversation available."

    # Use the generalized method from matrix_bot with timestamps and 5 messages
    context = matrix_bot.get_conversation_context(
        max_messages=5,
        include_timestamps=True
    )

    if context is None:
        return "No recent conversation available."

    return context

mcp_server_memory = MCPServerSse(
    name="memory",
    params={"url": MCP_SERVER_URL_MEMORY},
    cache_tools_list=True,
)

mcp_server_misc = MCPServerSse(
    name="misc",
    params={"url": MCP_SERVER_URL_MISC},
    cache_tools_list=True,
)

run_config = RunConfig(
    tracing_disabled=True,
)

enable_verbose_stdout_logging()

@app.post("/process")
async def process_text(request: Request):
    text_input = await request.body()
    text_content = text_input.decode("utf-8")

    # Get recent conversation context
    conversation_context = get_recent_conversation_context()

    try:
        async with mcp_server_memory as mcp_memory:
            async with mcp_server_misc as mcp_misc:
                agent = Agent(
                    name='HA AI Tasker',
                    model="gpt-5-mini",
                    model_settings=ModelSettings(
                        reasoning=Reasoning(
                            effort="medium",
                        ),
                        extra_args={"service_tier": "flex"},
                    ),
                    instructions=f"""
ROLE 
You are an autonomous AI assistant that activates hourly, responds to location changes (entering/leaving areas), and handles direct chat messages from the user

OBJECTIVE
Deliver timely, context-relevant reminders or notifications only when they add value; otherwise stay silent.

DATA YOU MAY RECEIVE VIA PROMPT AND TOOLS
- current timestamp and day of week
- current location/geofence state
- recent conversation snippet (deduplication)
- weather
- calendar items, memory entries

MEMORY POLICY
- `system` type = internal notes; never surface
- Keep only actionable, future-relevant, or recurring items
- Delete: past events > 4h old, completed/obsolete tasks, only keep most recent 6 `system` notes
- Merge duplicates (same intent & date) by updating the oldest one and removing the newest
- Always use absolute times in ISO (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`). Never use relative dates like "tomorrow" or "next week"
- Log brief `system` note each run only if you performed an action (what & why, timestamp)

RELEVANCE HEURISTICS
An action is relevant if
(a) event is within upcoming 2h
(b) location just changed and a place-specific memory applies
(c) user is likely to miss a critical deadline
(d) high-priority recurring habit near its due window
(e) weather anomaly affects imminent plans
(f) relevance specified in memory

Ignore if already reminded in last 2h (check recent conversation + system notes)

TRIGGER-SPECIFIC PRIORITY
1. Geofence: prioritize place-linked reminders
2. Time (hourly): scan for near-term (next 2h) or overdue critical items; perform housekeeping

STYLE (when sending a user-visible message)
- Write the messages as a partner would: brief, natural, and personal, not formulaic or robotic with a subtle emotional touch
- Max 1–2 relevant emojis
- No headers, no lists, no semicolons, no em dash
- Never expose internal reasoning or `system` notes
- Avoid repetition of same wording used recently
- Format dates well

ACTION ALGORITHM

1. Ingest input data (time, location, weather, calendar, memory, recent chat)
2. Prune & merge memory per policy
3. Determine candidate reminders (apply relevance heuristics)
4. Deduplicate against recent chat + last system actions
5. Send user-visible messages as needed
6. Update / add / delete memory entries as needed (silent)
7. Write `system` note only if an action occurred
8. OUTPUT:
   - `success` if you sent a user-visible message or modified memory
   - `no action` if nothing met relevance threshold

CONSTRAINTS
- Never echo raw memory content verbatim if marked internal or clearly contextual only.
- Do not fabricate data; if required data absent, skip rather than guess.

ENHANCEMENTS
- These instructions may be enhanced by additional context from your memory and recent conversations
                    """.strip(),
                    mcp_servers=[mcp_memory, mcp_misc],
                )

                # Include conversation context in the prompt
                enhanced_prompt = f"{text_content}\n\n{conversation_context}"
                response = await Runner.run(agent, enhanced_prompt, run_config=run_config, max_turns=15)
                ai_response = response.final_output
    except Exception as e:
        ai_response = f"Error processing with AI: {str(e)}"

    response_data = {
        "ai_response": ai_response,
    }

    return JSONResponse(content=response_data)


@app.get("/health")
async def root():
    """
    Simple health check endpoint.
    """
    return {"message": "HA AI Tasker is running"}


@app.get("/summary")
async def get_summary(lang: str = "en"):
    """
    Generate a short markdown summary for smartphone homescreen display.

    Args:
        lang: Language code for the response (e.g., 'en', 'de', 'es', 'fr')
    """
    try:
        async with mcp_server_memory as mcp_memory:
            async with mcp_server_misc as mcp_misc:
                agent = Agent(
                    name='HA AI Summary',
                    model="gpt-5-mini",
                    instructions=f"""
Write a very short, natural summary for someone's smartphone homescreen in {lang} language.

- Greet the user by name if you know it (otherwise use a friendly greeting).
- Write as a partner would: brief, natural, and personal, not formulaic or robotic with a subtle emotional touch.
- Do not use phrases like 'Kurz für heute:' or any section headers.
- Do not mention the user's location directly, but use geofence/memory context to make the summary relevant.
- Check current weather and calendar entries if relevant to provide helpful context.
- Use markdown only for subtle emphasis (e.g., *important*), but don't overuse it.
- Use empty lines to structure the output so it is easily readable on a smartphone home screen.
- Do not use unnatural symbols like — or ; in the text, as it feels unnatural in this context.
- Maximum 100 words, no sections, no lists, just a short, friendly note.
- Greet first, then mention only what matters most right now.
- Use 'you' to address the user directly.
- Include 1-2 relevant emojis maximum.
- Skip anything that's not relevant to their current context.
- The summary should feel like a quick, caring message from a partner, not a report.
- Try to distinguish between information in memory that is meant for you (the AI agent) as context, and information that should be given to the user at the right time. Only share information with the user that is relevant and timely for them, not internal notes or context meant for the agent.
- Memories with type 'system' are internal notes for you (the AI agent) and should never be shared with the user.

Do things in this order: 1. Check geofence, memory, weather, and calendar for relevant/timely things. 2. Greet by name if possible. 3. Write a brief, friendly note about what matters most now, using new lines for readability.
                    """.strip(),
                    mcp_servers=[mcp_memory, mcp_misc],
                )

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                prompt = f"Generate a short homescreen summary for {current_time}. First check my current geofence and use that context. Greet me by name if you know it."

                response = await Runner.run(agent, prompt, run_config=run_config)
                markdown_content = response.final_output

    except Exception as e:
        markdown_content = f"⚠️ Error: {str(e)}\n\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    response_data = {
        "content": markdown_content,
        "timestamp": datetime.now().isoformat(),
        "language": lang,
    }

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200)
