# HA AI Tasker

A smart AI assistant system that integrates with Home Assistant and Matrix to provide contextual notifications, memory management, and conversational chat capabilities.

## Features

### 🤖 FastAPI Web Server
- **Process Endpoint** (`POST /process`): Autonomous AI agent triggered periodically or by events
- **Summary Endpoint** (`GET /summary`): Generate smartphone homescreen summaries in multiple languages
- **Health Check** (`GET /health`): Service health monitoring

### 💬 Matrix Chat Bot
- Real-time conversational AI assistant in Matrix rooms
- Memory-powered responses with conversation history
- Room-specific filtering and message timestamp filtering
- Graceful handling of MCP server availability

### 🧠 Memory & Context Management
- Persistent memory storage via MCP (Model Context Protocol) servers
- User and system memory types for different data purposes
- Weather and calendar integration for contextual responses
- Geofence-aware location context

## Architecture

The system consists of:
- **FastAPI Application**: Web API endpoints for automation triggers
- **Matrix Bot**: Interactive chat assistant
- **MCP Servers**: Memory and miscellaneous tool servers
- **AI Agent**: OpenAI-powered decision making and response generation

## Installation

### Prerequisites
- Python 3.13+
- Poetry for dependency management
- Access to OpenAI API
- Matrix homeserver account
- MCP servers (memory and misc)

### Setup

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Configure environment variables:**
   ```bash
   # OpenAI Configuration
   export OPENAI_API_KEY=your-openai-api-key

   # Matrix Bot Configuration
   export MATRIX_HOMESERVER_URL=https://matrix.example.com
   export MATRIX_USERNAME=@your-bot:matrix.example.com
   export MATRIX_PASSWORD=your-bot-password
   export MATRIX_ROOM_ID=!room-id:matrix.example.com

   # MCP Server URLs
   export MCP_SERVER_URL_MEMORY=http://localhost:8300/sse
   export MCP_SERVER_URL_MISC=http://localhost:8100/sse
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8200
   ```

## Usage

### Autonomous AI Agent

Send POST requests to `/process` to trigger the AI agent:

```bash
curl -X POST http://localhost:8200/process \
  -H "Content-Type: text/plain" \
  -d "trigger event or context"
```

The agent will:
1. Check memory, time, location, weather, and calendar
2. Evaluate relevance and importance
3. Take appropriate actions (reminders, notifications, memory updates)
4. Output 'success' or 'no action'

### Homescreen Summary

Get a personalized summary for smartphone display:

```bash
curl "http://localhost:8200/summary?lang=en"
```

Supported languages: `en`, `de`, `es`, `fr`, etc.

### Matrix Chat

The bot automatically joins the configured Matrix room and responds to messages with:
- Conversation context awareness
- Memory-powered personalization
- Natural, partner-like communication style
- Emoji support (1-2 per message)

## Configuration

### AI Behavior

The system uses distinct instruction sets for different contexts:

- **Autonomous Agent**: Focuses on proactive notifications and memory management
- **Summary Generator**: Creates concise, friendly homescreen updates
- **Matrix Chat Bot**: Provides conversational responses with memory integration

### Memory Management

- **User memories**: Information about the user for contextual responses
- **System memories**: Internal notes for the AI agent (12-hour retention)
- **ISO date format**: All memory dates stored in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format

### Message Filtering

The Matrix bot implements several filters:
- **Room filtering**: Only responds in the configured room
- **Timestamp filtering**: Ignores messages sent before bot startup
- **Self-message handling**: Tracks own messages but doesn't respond to them

## Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t ha-ai-tasker .

# Run container
docker run -p 8200:8200 \
  -e OPENAI_API_KEY=your-key \
  -e MATRIX_HOMESERVER_URL=https://matrix.example.com \
  -e MATRIX_USERNAME=@bot:matrix.example.com \
  -e MATRIX_PASSWORD=bot-password \
  -e MATRIX_ROOM_ID=!room:matrix.example.com \
  -e MCP_SERVER_URL_MEMORY=http://memory-server:8300/sse \
  -e MCP_SERVER_URL_MISC=http://misc-server:8100/sse \
  ha-ai-tasker
```

## Dependencies

- **FastAPI**: Web framework
- **OpenAI Agents**: AI agent framework with MCP support
- **matrix-nio**: Matrix protocol client
- **uvicorn**: ASGI server

## Error Handling

- **MCP Server Unavailable**: Matrix bot silently skips message processing
- **Task Exceptions**: Comprehensive async task error handling
- **Graceful Degradation**: System continues operating with reduced functionality

## Contributing

The system is designed for:
- Home Assistant automation integration
- Personal AI assistant use cases
- Event-driven notification systems
- Conversational AI applications

## License

MIT License
