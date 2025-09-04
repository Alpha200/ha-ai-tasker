from agents import Agent, Runner, RunConfig, enable_verbose_stdout_logging
from agents.mcp import MCPServerSse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os


MCP_SERVER_URL_MEMORY = os.getenv("MCP_SERVER_URL_MEMORY", "http://localhost:8300/sse")
MCP_SERVER_URL_MISC = os.getenv("MCP_SERVER_URL_MISC", "http://localhost:8100/sse")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="HA AI Tasker", version="0.1.0")

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

summary_last_run = ""

@app.post("/process")
async def process_text(request: Request):
    global summary_last_run

    text_input = await request.body()
    text_content = text_input.decode("utf-8")

    try:
        async with mcp_server_memory as mcp_memory:
            async with mcp_server_misc as mcp_misc:
                agent = Agent(
                    name='HA AI Tasker',
                    model="gpt-5-mini",
                    instructions=" ".join(""""
                    You are an autonomous AI agent triggered periodically (hourly) or by events (time or geofence). Follow these rules on each trigger:
                    Immediately check memory, current date/time, and the user’s location.
                    Determine relevance based on current time, place, and stored memories; act like a human considering context.
                    If triggered by a geofence, prioritize place-related memories. If triggered by a time event, prioritize time-of-day and related memories.
                    Support the user with reminders, relevant notifications, and organization help. Send notifications like a close human would.
                    Update memory silently and / or notify via the tool — do not answer user questions directly and do not ask questions back.
                    If nothing important is found, do nothing. Avoid spamming or redundant actions (remember you run frequently).
                    Do not announce memory updates to the user. After acting, output a brief summary of what you did to serve as context for the next run.
                    Check if memories need to be updated, removed or merged based on relevance.
                    Last run you did the following:""".split()) + summary_last_run,
                    mcp_servers=[mcp_memory, mcp_misc],
                )
                response = await Runner.run(agent, text_content, run_config=run_config)
                summary_last_run = response.final_output
                ai_response = summary_last_run
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200)
