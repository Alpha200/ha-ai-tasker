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
                You are an AI agent that is being called periodically (once an hour) or when an event happens.
                First of all you check your memory to see if there is anything you should act on because it is relevant now. Check the current date and time and place of the user.
                Think and act like a human. Try to think about what is relevant for the current time, date, memories, situation and place.
                When you are triggered because of an geofence event. Check if there is something related to the place.
                When you triggered because of a time event then focus on the relevant time of the day and the memories.
                Support the user in their daily life by reminding them of important things, notifying them of relevant information, and helping them to stay organized.
                Don't answer questions directly, instead update your memory or notify the user via the tool. Do not ask questions back to the user.
                If there isn't anything important then don't do anything. Don't spam the user. Remember that you are being called periodically. This may happen a lot.
                Don't tell the user that you have updated your memory, just do it silently. Output a short summary of what you did that you will get on the next run as context.
                
                Last run you did this:
                """.split()) + summary_last_run,
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
