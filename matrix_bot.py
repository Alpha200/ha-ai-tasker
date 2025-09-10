import asyncio
from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse, SyncResponse
from agents import Agent, Runner, RunConfig
from agents.mcp import MCPServerSse
from datetime import datetime
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatrixChatBot:
    def __init__(self, homeserver: str, user_id: str, password: str, room_id: str, mcp_memory_url: str, system_username: str = None):
        self.homeserver = homeserver
        self.user_id = user_id
        self.password = password
        self.room_id = room_id
        self.system_username = system_username
        self.client = AsyncClient(self.homeserver, self.user_id)

        # Store the start time to filter out old messages
        self.start_time = datetime.now()

        # Store conversation history (last 10 messages)
        self.conversation_history = deque(maxlen=10)

        # MCP server setup with error handling
        try:
            self.mcp_memory = MCPServerSse(
                name="memory",
                params={"url": mcp_memory_url},
                cache_tools_list=True,
            )
            self.mcp_available = True
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            self.mcp_memory = None
            self.mcp_available = False

        self.run_config = RunConfig(tracing_disabled=True)

    def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        # Create async wrapper to handle the actual processing with proper error handling
        task = asyncio.create_task(self._handle_message(room, event))
        # Add error handling to prevent unhandled task exceptions
        task.add_done_callback(self._task_done_callback)
        return task

    def _task_done_callback(self, task):
        """Handle completed tasks and log any exceptions"""
        try:
            if task.exception():
                logger.error(f"Task failed with exception: {task.exception()}")
        except Exception as e:
            logger.error(f"Error in task done callback: {e}")

    async def _handle_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        try:
            # Only process messages from the specified room
            if room.room_id != self.room_id:
                return

            # Filter out messages that were sent before the bot started
            if event.server_timestamp < self.start_time.timestamp() * 1000:
                return

            logger.info(f"Received message from {event.sender} in {room.room_id}: {event.body}")

            # Add message to history (including our own messages from other apps)
            self.conversation_history.append({
                "sender": event.sender,
                "message": event.body,
                "timestamp": datetime.now().isoformat()
            })

            # Only respond if message is not from us
            if event.sender == self.client.user_id:
                return

            # Build conversation context
            history_text = self._build_conversation_context()

            try:
                # Process with AI - only if MCP is available
                if not self.mcp_available or not self.mcp_memory:
                    logger.info("MCP server not available, skipping message processing")
                    return

                async with self.mcp_memory as mcp_memory:
                    agent = Agent(
                        name='Matrix Chat Bot',
                        model="gpt-4o-mini",
                        instructions=f"""
You are a helpful AI assistant in a Matrix chat room.

- Check memory for relevant context about the user when needed.
- Determine relevance based on stored memories and conversation context; act like a human considering context.
- Help the user with questions, conversations, and organization when asked.
- Write responses as a partner would: brief, natural, and personal, not formulaic or robotic with a subtle emotional touch. Include 1-2 relevant emojis maximum when appropriate.
- Do not use phrases like 'Kurz für heute:'. Format dates well. Do not use technical stuff.
- Update memory silently when you learn important information about the user. Do not announce memory updates.
- Answer user questions directly and engage in natural conversation.
- Do not use unnatural symbols like — or ; in the text, as it feels unnatural in this context.
- Use memory entries with type 'system' to store internal notes for yourself that should not be shared with the user. Keep it brief with timestamps.
- Try to distinguish between information in memory that is meant for you (the AI agent) as context, and information that should be given to the user at the right time.
- When storing relevance dates in memories, always use ISO format dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS), not relative dates like "tomorrow" or "next week".
- Respond naturally to the user's messages based on the conversation history.
- Keep responses conversational and helpful.
- Use the memory tool when appropriate to remember important information about the user.
- Be concise but friendly.
- Consider the full conversation context when responding.

Do things in this order: 1. Check memory for relevant context about the user. 2. Evaluate the user's message and respond naturally. 3. Update memory if you learn something important about the user.
                        """.strip(),
                        mcp_servers=[mcp_memory],
                    )

                    response = await Runner.run(agent, history_text, run_config=self.run_config)
                    ai_response = response.final_output

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                return  # Do nothing on error

            # Send response
            await self.client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": ai_response
                }
            )

            logger.info(f"Sent response to {room.room_id}")

        except Exception as e:
            logger.error(f"Unhandled error in _handle_message: {e}")
            # Send a simple error message to the room
            try:
                await self.client.room_send(
                    room_id=room.room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body": "Sorry, I encountered an unexpected error."
                    }
                )
            except Exception as send_error:
                logger.error(f"Failed to send error message: {send_error}")

    def get_conversation_context(self, max_messages: int = 10, include_timestamps: bool = False) -> str | None:
        """Build conversation context from recent messages

        Args:
            max_messages: Maximum number of recent messages to include
            include_timestamps: Whether to include timestamps in the format [YYYY-MM-DD HH:MM]

        Returns:
            Conversation context string or None if no history available
        """
        if not self.conversation_history:
            return None

        # Build conversation context (last N messages)
        recent_messages = list(self.conversation_history)[-max_messages:] if len(self.conversation_history) > max_messages else list(self.conversation_history)
        context_lines = ["Recent conversation history:"]

        for msg in recent_messages:
            sender_name = msg["sender"].split(":")[0].replace("@", "")

            # Determine if sender is system or use real username
            if self.system_username and sender_name == self.system_username:
                role = "system"
            else:
                role = sender_name

            if include_timestamps:
                # Parse timestamp and format as date
                msg_time = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M")
                context_lines.append(f"[{msg_time}] {role}: {msg['message']}")
            else:
                context_lines.append(f"{role}: {msg['message']}")

        return "\n".join(context_lines)

    def _build_conversation_context(self) -> str:
        """Build conversation context from recent messages (legacy method for internal use)"""
        context = self.get_conversation_context(max_messages=10, include_timestamps=False)
        if context is None:
            return "No previous conversation history."
        return context + "\n\nPlease respond to the most recent message considering this conversation context."

    async def login_callback(self, response: LoginResponse):
        if isinstance(response, LoginResponse):
            logger.info(f"Logged in as {response.user_id}")
        else:
            logger.error(f"Login failed: {response}")

    async def sync_callback(self, response: SyncResponse) -> None:
        logger.info(f"Sync completed")

    async def start(self):
        # Login
        response = await self.client.login(self.password)
        if not isinstance(response, LoginResponse):
            logger.error(f"Failed to login: {response}")
            return

        # Add callbacks
        self.client.add_event_callback(self.message_callback, RoomMessageText)

        logger.info("Starting Matrix bot...")

        # Start syncing
        await self.client.sync_forever(timeout=30000)

    async def stop(self):
        await self.client.close()
