from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import openai


class AIProviderBase(ABC):
    """Abstract base class for AI providers"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Tuple[str, bool, Any]:
        """
        Generate a response from the AI model.

        Args:
            messages: List of conversation messages
            system_prompt: System prompt for the AI
            tools: Optional list of tools available to the AI

        Returns:
            Tuple of (response_text, has_tool_calls, raw_response)
        """
        pass

    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from the provider's response format"""
        pass

    @abstractmethod
    def format_tool_results(
        self, tool_results: List[Dict[str, Any]], response: Any
    ) -> Dict[str, Any]:
        """Format tool results in the provider's expected format"""
        pass


class AnthropicProvider(AIProviderBase):
    """Anthropic Claude API provider"""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Tuple[str, bool, Any]:
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        has_tool_calls = response.stop_reason == "tool_use"

        if has_tool_calls:
            return "", True, response
        else:
            return response.content[0].text, False, response

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic response"""
        tool_calls = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input,
                    }
                )
        return tool_calls

    def format_tool_results(
        self, tool_results: List[Dict[str, Any]], response: Any
    ) -> Dict[str, Any]:
        """Format tool results for Anthropic"""
        # Anthropic expects assistant message with tool use, then user message with results
        return {
            "assistant_content": response.content,
            "user_content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result["id"],
                    "content": result["result"],
                }
                for result in tool_results
            ],
        }


class OpenAIProvider(AIProviderBase):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key)
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def _convert_tools_to_openai_format(
        self, tools: Optional[List]
    ) -> Optional[List[Dict]]:
        """Convert Anthropic tool format to OpenAI function format"""
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _convert_messages_to_openai_format(
        self, messages: List[Dict[str, Any]], system_prompt: str
    ) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format, including system prompt"""
        openai_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            role = msg["role"]

            # If this is already an OpenAI-format message (has role + other fields), pass it through
            # This handles assistant messages with tool_calls and tool response messages
            if role == "tool" and "tool_call_id" in msg:
                # Tool response message - already in correct format
                openai_messages.append(msg)
                continue
            elif role == "assistant" and "tool_calls" in msg:
                # Assistant message with tool calls - already in correct format
                openai_messages.append(msg)
                continue

            content = msg.get("content", "")

            # Handle different content types
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Handle tool results or mixed content (Anthropic format)
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            # Convert Anthropic tool result to OpenAI format
                            openai_messages.append(
                                {
                                    "role": "function",
                                    "name": "tool_response",
                                    "content": item["content"],
                                }
                            )
                        elif hasattr(item, "type") and item.type == "tool_use":
                            # This is handled differently - will be in tool_calls
                            pass
                        elif item.get("type") == "text":
                            openai_messages.append(
                                {"role": role, "content": item.get("text", "")}
                            )

        return openai_messages

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Tuple[str, bool, Any]:
        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai_format(tools)

        # Convert messages to OpenAI format
        openai_messages = self._convert_messages_to_openai_format(
            messages, system_prompt
        )

        api_params = {**self.base_params, "messages": openai_messages}

        if openai_tools:
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)

        # Check if there are tool calls
        has_tool_calls = (
            response.choices[0].message.tool_calls is not None
            and len(response.choices[0].message.tool_calls) > 0
        )

        if has_tool_calls:
            return "", True, response
        else:
            return response.choices[0].message.content or "", False, response

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI response"""
        tool_calls = []
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                import json

                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments),
                    }
                )
        return tool_calls

    def format_tool_results(
        self, tool_results: List[Dict[str, Any]], response: Any
    ) -> Dict[str, Any]:
        """Format tool results for OpenAI"""
        # OpenAI expects assistant message with tool calls, then tool messages with results
        # Use the actual tool_calls from the response to preserve arguments
        message = response.choices[0].message

        # Convert tool_calls to dict format for JSON serialization
        tool_calls_list = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls_list.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })

        return {
            "assistant_content": {
                "role": "assistant",
                "content": message.content,
                "tool_calls": tool_calls_list,
            },
            "tool_messages": [
                {
                    "role": "tool",
                    "tool_call_id": result["id"],
                    "content": result["result"],
                }
                for result in tool_results
            ],
        }


def create_ai_provider(provider_type: str, api_key: str, model: str) -> AIProviderBase:
    """Factory function to create AI provider instances"""
    if provider_type.lower() == "anthropic":
        return AnthropicProvider(api_key, model)
    elif provider_type.lower() == "openai":
        return OpenAIProvider(api_key, model)
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. Choose 'anthropic' or 'openai'"
        )
