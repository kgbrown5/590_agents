"""The course starter agent, independent of Harbor."""

import json
import subprocess
from pathlib import Path
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
)

from agents590.atif import (
    ATIFFinalMetrics,
    ATIFMetrics,
    ATIFRecorder,
    ATIFToolCall,
    JSONValue,
)
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class KgbroAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    system_prompt = (
        "You are a coding agent. Use the provided tools to inspect and edit files "
        "or run commands, then give the user a concise summary."
    )

    tools: list[FunctionToolParam] = [
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
                "additionalProperties": False,
            },
        },
    ]

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        settings = self.provider_settings(context)

        async with settings.create_async_client() as client:
            return await self._loop(
                client=client,
                model=settings.model,
                instruction=instruction,
                context=context,
                recorder=ATIFRecorder.from_environment(agent="kgbro", model=settings.model),
            )

    async def _loop(
        self,
        client: AsyncOpenAI,
        model: str,
        instruction: str,
        context: RunContext,
        recorder: ATIFRecorder,
    ) -> str:

        recorder.add("system", self.system_prompt)
        recorder.add("user", instruction)

        conversation: list[ResponseInputItemParam] = [
            {
                "role": "user",
                "content": instruction,
            },
        ]

        for _ in range(50):
            response = await client.responses.create(
                model=model,
                instructions=self.system_prompt,
                tools=self.tools,
                input=conversation,
                store=False,
                background=False,
            )

            conversation.extend(
                cast(ResponseInputItemParam, item.model_dump(exclude_none=True))
                for item in response.output
            )

            tool_calls = [
                item for item in response.output if isinstance(item, ResponseFunctionToolCall)
            ]

            if not tool_calls:
                self._finalize_trajectory(recorder)
                return response.output_text

            outputs: list[tuple[str, str]] = []
            for tool in tool_calls:
                result = self._run_tool(tool, context.cwd)
                outputs.append((tool.call_id, result))
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool.call_id,
                        "output": result,
                    }
                )

            self._record_response(response, tool_calls, outputs, recorder)

        self._finalize_trajectory(recorder)
        return "Agent turn limit reached."

    def _record_response(
        self,
        response: Response,
        tool_calls: list[ResponseFunctionToolCall],
        outputs: list[tuple[str, str]],
        recorder: ATIFRecorder,
    ) -> None:
        fields: dict[str, JSONValue] = {
            "model_name": response.model,
        }
        if tool_calls:
            calls: list[ATIFToolCall] = [
                {
                    "tool_call_id": call.call_id,
                    "function_name": call.name,
                    "arguments": self._arguments_for_atif(call.arguments),
                }
                for call in tool_calls
            ]
            fields["tool_calls"] = cast(JSONValue, calls)
            fields["observation"] = {
                "results": [
                    {"source_call_id": call_id, "content": content} for call_id, content in outputs
                ]
            }

        usage = response.usage
        if usage is not None:
            metrics: ATIFMetrics = {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "cached_tokens": usage.input_tokens_details.cached_tokens,
            }
            fields["metrics"] = cast(JSONValue, metrics)

        recorder.add("agent", response.output_text, **fields)

    def _arguments_for_atif(self, arguments: str) -> JSONValue:
        try:
            parsed: object = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
            return cast(JSONValue, parsed)
        return {"raw": arguments}

    def _finalize_trajectory(self, recorder: ATIFRecorder) -> None:
        metrics = [step["metrics"] for step in recorder.document["steps"] if "metrics" in step]
        final_metrics: ATIFFinalMetrics = {
            "total_prompt_tokens": sum(item["prompt_tokens"] for item in metrics),
            "total_completion_tokens": sum(item["completion_tokens"] for item in metrics),
            "total_cached_tokens": sum(item["cached_tokens"] for item in metrics),
            "total_steps": len(recorder.document["steps"]),
        }
        recorder.document["final_metrics"] = final_metrics
        recorder.write()

    def _run_tool(self, tool: ResponseFunctionToolCall, cwd: Path) -> str:
        try:
            parsed_arguments: object = json.loads(tool.arguments)
        except json.JSONDecodeError:
            return "Error: Tool arguments were not valid JSON."
        if not isinstance(parsed_arguments, dict):
            return "Error: Tool arguments must be a JSON object."
        arguments = cast(dict[str, object], parsed_arguments)

        if tool.name == "read_file":
            filename = arguments.get("filename")
            if not isinstance(filename, str) or not filename:
                return "Error: No filename provided."
            try:
                path = self._resolve_path(filename, cwd)
                return path.read_text(encoding="utf-8")
            except (OSError, ValueError) as error:
                return f"Error reading file: {error}"

        elif tool.name == "write_file":
            filename = arguments.get("filename")
            contents = arguments.get("contents")
            if not isinstance(filename, str) or not filename or not isinstance(contents, str):
                return "Error: Filename or contents not provided."
            try:
                path = self._resolve_path(filename, cwd)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
                return f"Successfully wrote to {filename}."
            except (OSError, ValueError) as error:
                return f"Error writing to file: {error}"

        elif tool.name == "run_command":
            command = arguments.get("command")
            if not isinstance(command, str) or not command:
                return "Error: No command provided."
            try:
                result = subprocess.run(
                    command, shell=True, cwd=cwd, capture_output=True, text=True
                )
                return (
                    f"exit_code: {result.returncode}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            except OSError as error:
                return f"Error running command: {error}"

        else:
            return f"Unknown tool: {tool.name}"

    def _resolve_path(self, filename: str, cwd: Path) -> Path:
        root = cwd.resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Path is outside the working directory.")
        return path
