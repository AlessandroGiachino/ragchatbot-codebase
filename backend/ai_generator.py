from typing import Any, Dict, List, Optional

from ai_providers import AIProviderBase


class AIGenerator:
    """Handles interactions with AI providers (Anthropic Claude or OpenAI) for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews

Tool Usage Guidelines:
- Use content search for detailed questions about specific topics or lessons
- Use course outline tool for questions about course structure, lesson titles, or complete course overviews
- **You can make up to 2 rounds of tool calls to gather comprehensive information**
- Use multiple rounds for complex queries that require information gathering then refinement
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Course Outline Responses:
When using the course outline tool, always include:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles
- Present information in a clear, structured format

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, provider: AIProviderBase):
        self.provider = provider

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential rounds of tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Start with initial messages
        messages = [{"role": "user", "content": query}]

        # Execute up to 2 rounds of tool calling
        for round_num in range(2):
            # Get response from the AI provider
            response_text, has_tool_calls, raw_response = (
                self.provider.generate_response(
                    messages=messages,
                    system_prompt=system_content,
                    tools=tools,
                )
            )

            # Handle tool execution if needed
            if has_tool_calls and tool_manager:
                messages, should_continue = self._handle_tool_execution(
                    raw_response, messages, tool_manager
                )
                if not should_continue:
                    break
            else:
                # No tool use, return direct response
                return response_text

        # After max rounds, make final call without tools to get response
        final_text, _, _ = self.provider.generate_response(
            messages=messages,
            system_prompt=system_content,
            tools=None,
        )
        return final_text

    def _handle_tool_execution(self, initial_response, messages: List, tool_manager):
        """
        Handle execution of tool calls and update message history.

        Args:
            initial_response: The response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated_messages, should_continue)
        """
        # Extract tool calls using provider-specific method
        tool_calls = self.provider.extract_tool_calls(initial_response)

        # Execute all tool calls and collect results
        tool_results = []
        for tool_call in tool_calls:
            try:
                tool_result = tool_manager.execute_tool(
                    tool_call["name"], **tool_call["input"]
                )

                tool_results.append(
                    {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "result": tool_result,
                    }
                )
            except Exception as e:
                # Tool execution failed, stop rounds
                tool_results.append(
                    {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "result": f"Error: Tool execution failed - {str(e)}",
                    }
                )
                # Format and add tool results, then signal to stop
                formatted = self.provider.format_tool_results(
                    tool_results, initial_response
                )
                self._append_formatted_results(messages, formatted)
                return messages, False

        # Format tool results in provider-specific format
        formatted = self.provider.format_tool_results(tool_results, initial_response)
        self._append_formatted_results(messages, formatted)

        # Continue with next round
        return messages, True

    def _append_formatted_results(self, messages: List, formatted: Dict[str, Any]):
        """Append formatted tool results to messages based on provider format"""
        if "assistant_content" in formatted:
            assistant_content = formatted["assistant_content"]
            # Check if it's already a complete message dict (OpenAI format)
            if isinstance(assistant_content, dict) and "role" in assistant_content:
                messages.append(assistant_content)
            else:
                # It's just content, wrap it in a message (Anthropic format)
                messages.append({"role": "assistant", "content": assistant_content})
        if "user_content" in formatted:
            messages.append({"role": "user", "content": formatted["user_content"]})
        if "tool_messages" in formatted:
            for tool_msg in formatted["tool_messages"]:
                messages.append(tool_msg)
