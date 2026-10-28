"""The course starter agent, independent of Harbor."""

from openai.types.responses import ResponseInputParam, ToolParam

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class KgbroAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    tools: list[ToolParam] = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to be read",
                    },
                },
                "required": ["filename"],
            },
        },
        {
            "type": "function",
            "name": "write_file",
            "description": "Write to a file.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to be written to",
                    },
                    "contents": {
                        "type": "string",
                        "description": "Text to write to the file",
                    },
                },
                "required": ["filename", "contents"],
            },
        },
        {
            "type": "function",
            "name": "run_command",
            "description": "Execute a shell command to inspect files, modify code, or run tests.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    ]

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        # Start with settings = self.provider_settings(context), then open
        # settings.create_async_client() and use model=settings.model.
        # Send Responses requests with store=False and background=False.
        # Run shell tools with context.cwd as their explicit working directory.
        settings = self.provider_settings(context)
        client = settings.create_async_client()

        messages: ResponseInputParam = [
            {
                "role": "user",
                "content": instruction,
            },
        ]

        while True:
            response = await client.responses.create(
                model=settings.model,
                tools=self.tools,
                input=messages,
                store=False,
                background=False,
            )

            messages.append(response.output)

            tool_calls = [item for item in response.output if item.type == "function_call"]

            if not tool_calls:
                break

        return response.output_text

        # raise NotImplementedError("Implement the UNC Azure model/tool loop in KgbroAgent.run().")
