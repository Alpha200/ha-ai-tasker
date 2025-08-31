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

last_notification_text = "No previous notifications."

@app.post("/process")
async def process_text(request: Request):
    global last_notification_text

    text_input = await request.body()
    text_content = text_input.decode("utf-8")

    try:
        async with mcp_server_memory as mcp_memory:
            async with mcp_server_misc as mcp_misc:
                agent = Agent(
                    name='HA AI Tasker',
                    model="gpt-5-mini",
                    instructions="""
                You are an AI agent that is being called periodically (once an hour) or when an event happens.
                First of all you check your memory to see if there is anything the user should be reminded of. 
                Don't remind the user for time based things when you have triggered because of a geofence. Then check for memories for that place.
                Don't check for place based memories when you have triggered because of a time event.
                Don't answer questions directly, instead update the memory or notify the user about important information.
                If there isn't anything important then don't do anything. Remember that you are being called periodically. This may happen a lot.
                Don't tell the user that you have updated your memory, just do it silently. Your last response should be the notification text you have sent to the user.
                Respond with "No response generated" if you didn't notify the user.
                
                Your last notification to the user was: "
                """.strip() + last_notification_text,
                    mcp_servers=[mcp_memory, mcp_misc],
                )
                response = await Runner.run(agent, text_content, run_config=run_config)

                ai_response = response.final_output if response.final_output else "No response generated"
                last_notification_text = response.final_output
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
