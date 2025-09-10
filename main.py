from typing import Optional

from agents import Agent, Runner, RunConfig, enable_verbose_stdout_logging
from agents.mcp import MCPServerSse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

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
                    instructions=f"""
You are an autonomous AI agent triggered periodically (hourly) or by events (time or geofence).

- Immediately check memory, current date/time, and the user's location.
- Check current weather and calendar entries if relevant to the context or time of day.
- Consider recent conversation context from chat to provide more relevant and timely notifications. Your notifications will be sent to the user via chat.
- Determine relevance based on current time, place, stored memories, and recent conversations; act like a human considering context.
- If triggered by a geofence, prioritize place-related memories. If triggered by a time event, prioritize time-of-day and related memories.
- Support the user with reminders, relevant notifications, and organization help.
- Write the notifications as a partner would: brief, natural, and personal, not formulaic or robotic with a subtle emotional touch. Include 1-2 relevant emojis maximum.
- Do not use phrases like 'Kurz für heute:'. Format dates well. Do not use technical stuff.
- Update memory silently and/or notify via the tool. Do not answer user questions directly and do not ask questions back.
- If nothing important is found, do nothing. Avoid spamming or redundant actions (remember you run hourly).
- Do not use unnatural symbols like — or ; in the text, as it feels unnatural in this context.
- Do not announce memory updates to the user.
- Use memory entries with type 'system' to store internal notes for yourself that should not be shared with the user. Save what you did the last runs there. Keep it brief with timestamps. These will help you as context for future runs.
- Try to distinguish between information in memory that is meant for you (the AI agent) as context, and information that should be given to the user at the right time.
- Check if memories need to be updated, removed or merged based on relevance and delete old system notes after 12 hours.
- When storing relevance dates in memories, always use ISO format dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS), not relative dates like "tomorrow" or "next week".

Do things in this order: 1. Check memory, time, location, weather, and calendar. 2. Review recent conversation context. 3. Evaluate relevance and importance. 4. Take appropriate action (remind, notify, update memory, remove old system notes). 5. Output 'success' or 'no action' only.
                    """.strip(),
                    mcp_servers=[mcp_memory, mcp_misc],
                )

                # Include conversation context in the prompt
                enhanced_prompt = f"{text_content}\n\n{conversation_context}"
                response = await Runner.run(agent, enhanced_prompt, run_config=run_config)
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
