"""
title: OpenRouter Integration for OpenWebUI
version: 0.4.6
description: Integration with OpenRouter for OpenWebUI with Free Model Filtering and optional field improvements
author: kevarch
author_url: https://github.com/kevarch
contributors:
    Eloi Marques da Silva (https://github.com/eloimarquessilva),
    Scythe Eden (https://github.com/DarkEden-coding),
    Kevin Pham (https://github.com/konmeo)
credits: rburmorrison (https://github.com/rburmorrison), Google Gemini Pro 2.5, Claude 4 Sonnet
license: MIT

Changelog:
- Version 0.4.6:
  * Contribution by Kevin Pham
  * Changed FREE_ONLY with MAX_INPUT_PRICE parameter to optionally filter and set set model price cap
  * Changed SHOW_USAGE_STATS to display extended usage statistics in info tooltip instead of at the end of responses
  * Added SHOW_MODEL_DESCRIPTION parameter to display model description and statistics
- Version 0.4.5:
  * Contribution by Scythe Eden
  * Added _sanitize_messages_for_notifications function to remove notification lines from messages before sending to the model.
  * Fixed streaming not working for thinking.
  * Fixed cancel button not working.
  * Fixed double output issue.
- Version 0.4.4:
  * Contribution by Scythe Eden
  * Added SHOW_USAGE_STATS parameter to display token usage and cost information at the end of responses.
  * Implemented usage statistics tracking for both streaming and non-streaming modes.
  * Added formatted display showing input, reasoning, output tokens and total cost in dollars.
- Version 0.4.3:
  * Contribution by Scythe Eden
  * Added MODEL_PROVIDER_BLACKLIST parameter and logic to filter models by specific model IDs.
- Version 0.4.2:
  * Contribution by Scythe Eden
  * Added MODEL_WHITELIST parameter to filter models by specific model IDs.
  * Added pricing information display in model names format: ($input, $output) $total (per 1M tokens).
  * Added REASONING_EFFORT parameter to control reasoning token allocation (high, medium, low).
  * Implemented dynamic reasoning effort detection from the first word of user messages.
  * Added a Markdown-formatted effort notification at the beginning of the returned message.
  * Automatically removes the effort keyword from the user's message after detection.
- Version 0.4.1:
  * Contribution by Eloi Marques da Silva
  * Added FREE_ONLY parameter to optionally filter and display only free models in OpenWebUI.
  * Changed MODEL_PREFIX and MODEL_PROVIDERS from required (str) to optional (Optional[str]), allowing null values.
"""

import re
import requests
import json
import traceback  # Import traceback for detailed error logging
from typing import Optional, List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from datetime import datetime


# --- Helper function for citation text insertion ---
def _insert_citations(text: str, citations: list[str]) -> str:
    """
    Replace citation markers [n] in text with markdown links to the corresponding citation URLs.

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in the text

    Returns:
        Text with citation markers replaced with markdown links
    """
    if not citations or not text:
        return text

    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        try:
            num = int(match_obj.group(1))
            if 1 <= num <= len(citations):
                url = citations[num - 1]
                return f"[[{num}]]({url})"
            else:
                return match_obj.group(0)
        except (ValueError, IndexError):
            return match_obj.group(0)

    try:
        return re.sub(pattern, replace_citation, text)
    except Exception as e:
        print(f"Error during citation insertion: {e}")
        return text


# --- Helper function for formatting the final citation list ---
def _format_citation_list(citations: list[str]) -> str:
    """
    Formats a list of citation URLs into a markdown string.

    Args:
        citations: A list of citation URLs.

    Returns:
        A formatted markdown string (e.g., "\n\n---\nCitations:\n1. url1\n2. url2")
        or an empty string if no citations are provided.
    """
    if not citations:
        return ""

    try:
        citation_list = [f"{i+1}. {url}" for i, url in enumerate(citations)]
        return "\n\n---\nCitations:\n" + "\n".join(citation_list)
    except Exception as e:
        print(f"Error formatting citation list: {e}")
        return ""


# --- Helper function for pricing formatting ---
def _format_pricing(model_data: dict) -> tuple:
    """
    Format pricing information from model data.

    Args:
        model_data: Model data from OpenRouter API

    Returns:
        Formatted pricing like: $0.15, $0.60, $0.75
    """
    try:
        pricing = model_data.get("pricing", {})
        if not pricing:
            return ""

        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            return ""

        # Convert from per-token to per-1M tokens for readability
        prompt_per_1m = max(-1, float(prompt_price) * 1_000_000)
        completion_per_1m = max(-1, float(completion_price) * 1_000_000)
        total_per_1m = prompt_per_1m + completion_per_1m

        return f"${prompt_per_1m:.2f}", f"${completion_per_1m:.2f}", f"${total_per_1m:.2f}"

    except (ValueError, TypeError, AttributeError) as e:
        print(
            f"Error formatting pricing for model {model_data.get('id', 'unknown')}: {e}"
        )
        return ""


# --- Helper function for effort detection and message processing ---
def _process_effort_from_message(
    messages: list, default_effort: str
) -> tuple[str, str]:
    """
    Process messages to detect effort level from first word and return effort + notification.

    Args:
        messages: List of messages from the request
        default_effort: Default effort level from settings

    Returns:
        Tuple of (effort_level, effort_notification_text)
    """
    if not messages:
        return default_effort, ""

    # Find the last user message
    last_user_message = None
    last_user_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_message = messages[i]
            last_user_index = i
            break

    if not last_user_message:
        return default_effort, ""

    # Extract content - handle both string and list formats
    content = last_user_message.get("content", "")
    if isinstance(content, list):
        # Find first text content
        text_content = ""
        for part in content:
            if part.get("type") == "text":
                text_content = part.get("text", "")
                break
        content = text_content

    if not isinstance(content, str) or not content.strip():
        return default_effort, ""

    # Check if first word is an effort level
    words = content.strip().split()
    if not words:
        return default_effort, ""

    first_word = words[0].lower()
    valid_efforts = ["high", "medium", "low"]

    if first_word in valid_efforts:
        # Remove the effort word from the message
        remaining_text = " ".join(words[1:]).strip()

        # Update the message content
        if isinstance(last_user_message["content"], list):
            # Update the text part in the list
            for part in last_user_message["content"]:
                if part.get("type") == "text":
                    part["text"] = remaining_text
                    break
        else:
            # Update string content
            messages[last_user_index]["content"] = remaining_text

        # Create effort notification using Markdown
        effort_notification = f"> *Reasoning set to {first_word}*\n\n"

        return first_word, effort_notification

    return default_effort, ""


# --- Helper function to remove notification lines from messages ---
def _sanitize_messages_for_notifications(messages: list) -> list:
    """Remove notification lines from messages before sending to the model.

    This strips lines like "> *Reasoning set to ...*" and
    "> *Ignoring providers for ...*" from both string and list-form message
    contents so they do not persist in model context in subsequent turns.

    Args:
        messages: Message list compatible with OpenAI chat format.

    Returns:
        Sanitized message list with notification lines removed.
    """
    if not isinstance(messages, list):
        return messages

    _USAGE_PATTERN = re.compile(r"\n\n---\n\*\*Usage\:\*\*.*?\$\d+\.\d{4}")

    def remove_notification_lines(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        lines = text.splitlines()
        filtered_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("> *Reasoning set to ") and stripped.endswith("*"):
                continue
            if stripped.startswith("> *Ignoring providers for ") and stripped.endswith("*"):
                continue
            filtered_lines.append(line)

        result = "\n".join(filtered_lines)

        # Remove the token usage strings strings
        result = re.sub(_USAGE_PATTERN, "", result)
        return result

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = remove_notification_lines(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part_text = part.get("text", "")
                    part["text"] = remove_notification_lines(part_text)
    return messages


# --- Main Pipe class ---
class Pipe:
    class Valves(BaseModel):
        # User-configurable settings
        OPENROUTER_API_KEY: str = Field(
            default="", description="Your OpenRouter API key (required)."
        )
        INCLUDE_REASONING: bool = Field(
            default=True,
            description="Request reasoning tokens from models that support it.",
        )
        REASONING_EFFORT: str = Field(
            default="medium",
            description="Default reasoning effort level: 'high' (~80% tokens), 'medium' (~50% tokens), 'low' (~20% tokens). Can be overridden by starting message with effort level.",
        )
        MODEL_PREFIX: Optional[str] = Field(
            default=None,
            description="Optional prefix for model names in Open WebUI (e.g., 'OR: ').",
        )
        # NEW: Configurable request timeout
        REQUEST_TIMEOUT: int = Field(
            default=90,
            description="Timeout for API requests in seconds.",
            gt=0,
        )
        MODEL_PROVIDERS: Optional[str] = Field(
            default=None,
            description="Comma-separated list of model providers to include or exclude. Leave empty to include all providers.",
        )
        INVERT_PROVIDER_LIST: bool = Field(
            default=False,
            description="If true, the above 'Model Providers' list becomes an *exclude* list instead of an *include* list.",
        )
        MODEL_WHITELIST: Optional[str] = Field(
            default=None,
            description="Comma-separated list of specific model IDs to include. If set, only these models will be available. Leave empty to use provider filtering instead.",
        )
        MODEL_PROVIDER_BLACKLIST: Optional[str] = Field(
            default=None,
            description="Model-specific provider blacklist in format 'model1:provider1,provider2;model2:provider3'. Example: 'gpt-4:openai;claude:anthropic,aws' would exclude OpenAI's GPT-4 and Anthropic/AWS versions of Claude models.",
        )
        ENABLE_CACHE_CONTROL: bool = Field(
            default=False,
            description="Enable OpenRouter prompt caching by adding 'cache_control' to potentially large message parts. May reduce costs for supported models (e.g., Anthropic, Gemini) on subsequent calls with the same cached prefix. See OpenRouter docs for details.",
        )
        MAX_INPUT_PRICE: Optional[float] = Field(
            default=None,
            description="Only show models that cost at most this amount per million input tokens.  Set to 0 for free models only.",
        )
        SHOW_PRICING: bool = Field(
            default=True,
            description="If true, show pricing information in model names (per 1M tokens).",
        )
        SHOW_USAGE_STATS: bool = Field(
            default=False,
            description="If true, show extended usage statistics (input, reasoning, output tokens and cost) in info tooltip.",
        )
        SHOW_MODEL_DESCRIPTION: bool = Field(
            default=False,
            description="If true, show model statistics (creation date, context size, and pricings) in model description.",
        )

    def __init__(self):
        self.type = "manifold"  # Specifies this pipe provides multiple models
        self.valves = self.Valves()
        if not self.valves.OPENROUTER_API_KEY:
            print("Warning: OPENROUTER_API_KEY is not set in Valves.")

    def pipes(self) -> List[dict]:
        """
        Fetches available models from the OpenRouter API.
        This method is called by OpenWebUI to discover the models this pipe provides.
        """
        if not self.valves.OPENROUTER_API_KEY:
            return [
                {"id": "error", "name": "Pipe Error: OpenRouter API Key not provided"}
            ]

        try:
            headers = {"Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}"}
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            models_data = response.json()
            raw_models_data = models_data.get("data", [])
            models: List[dict] = []

            # --- Whitelist Filtering Logic ---
            whitelist_str = (self.valves.MODEL_WHITELIST or "").strip()
            whitelist_models = set()
            if whitelist_str:
                whitelist_models = {
                    model_id.strip().lower()
                    for model_id in whitelist_str.split(",")
                    if model_id.strip()
                }

            # --- Provider Filtering Logic ---
            provider_list_str = (self.valves.MODEL_PROVIDERS or "").lower()
            invert_list = self.valves.INVERT_PROVIDER_LIST
            target_providers = {
                p.strip() for p in provider_list_str.split(",") if p.strip()
            }
            # --- End Filtering Logic ---

            for model in raw_models_data:
                model_id = model.get("id")
                if not model_id:
                    continue

                # Apply Whitelist Filtering (takes precedence over provider filtering)
                if whitelist_models:
                    if model_id.lower() not in whitelist_models:
                        continue
                else:
                    # Apply Provider Filtering only if no whitelist is set
                    if target_providers:
                        provider = (
                            model_id.split("/", 1)[0].lower()
                            if "/" in model_id
                            else model_id.lower()
                        )
                        provider_in_list = provider in target_providers
                        keep = (provider_in_list and not invert_list) or (
                            not provider_in_list and invert_list
                        )
                        if not keep:
                            continue

                # Apply Pricing Filtering
                if (
                    self.valves.MAX_INPUT_PRICE is not None
                    and float(model.get("pricing", {}).get("prompt", 0)) * 1000000
                    > self.valves.MAX_INPUT_PRICE
                ):
                    continue

                model_name = model.get("name", model_id)
                prefix = self.valves.MODEL_PREFIX or ""

                description = None

                prompt_price, completion_price, total_price = _format_pricing(model)

                # Add pricing information if enabled
                pricing_info = ""
                if self.valves.SHOW_PRICING:
                    pricing_info = f" ({prompt_price}, {completion_price}) {total_price}"

                if self.valves.SHOW_MODEL_DESCRIPTION:
                    description = (
                        f'({datetime.fromtimestamp(model.get("created")).strftime("%m/%Y")}: '
                        + f'{int(model.get("context_length", 0)) / 1000:,.0f}K '
                        + f'${prompt_price} / ${completion_price})\n\n'
                        + model.get("description", "")
                    )

                formatted_name = f"{prefix}{model_name}{pricing_info}"

                models.append({
                    "id": model_id,
                    "name": formatted_name,
                    "description": description
                })

            if not models:
                if self.valves.FREE_ONLY:
                    return [{"id": "error", "name": "Pipe Error: No free models found"}]
                elif whitelist_models:
                    return [
                        {
                            "id": "error",
                            "name": "Pipe Error: No models found matching the whitelist",
                        }
                    ]
                elif target_providers:
                    return [
                        {
                            "id": "error",
                            "name": "Pipe Error: No models found matching the provider filter",
                        }
                    ]
                else:
                    return [
                        {
                            "id": "error",
                            "name": "Pipe Error: No models found on OpenRouter",
                        }
                    ]

            return models

        except requests.exceptions.Timeout:
            print("Error fetching models: Request timed out.")
            return [{"id": "error", "name": "Pipe Error: Timeout fetching models"}]
        except requests.exceptions.HTTPError as e:
            error_msg = f"Pipe Error: HTTP {e.response.status_code} fetching models"
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
                if error_detail:
                    error_msg += f": {error_detail}"
            except json.JSONDecodeError:
                pass
            print(f"Error fetching models: {error_msg} (URL: {e.request.url})")
            return [{"id": "error", "name": error_msg}]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: Request failed: {e}")
            return [
                {
                    "id": "error",
                    "name": f"Pipe Error: Network error fetching models: {e}",
                }
            ]
        except Exception as e:
            print(f"Unexpected error fetching models: {e}")
            traceback.print_exc()
            return [{"id": "error", "name": f"Pipe Error: Unexpected error: {e}"}]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """
        Processes incoming chat requests. This is the main function called by OpenWebUI
        when a user interacts with a model provided by this pipe.

        Args:
            body: The request body, conforming to OpenAI chat completions format.

        Returns:
            Either a string (for non-streaming responses) or a generator/iterator
            (for streaming responses).
        """
        if not self.valves.OPENROUTER_API_KEY:
            return "Pipe Error: OpenRouter API Key is not configured."

        try:
            payload = body.copy()
            if "model" in payload and payload["model"] and "." in payload["model"]:
                payload["model"] = payload["model"].split(".", 1)[1]

            # --- Process effort detection from message ---
            effort_level = self.valves.REASONING_EFFORT.lower()
            effort_notification = ""

            if self.valves.INCLUDE_REASONING and "messages" in payload:
                effort_level, effort_notification = _process_effort_from_message(
                    payload["messages"], effort_level
                )

            # Store effort notification for response formatting
            self._effort_notification = effort_notification
            # --- End effort detection ---

            # --- Apply Cache Control Logic ---
            if self.valves.ENABLE_CACHE_CONTROL and "messages" in payload:
                try:
                    cache_applied = False
                    messages = payload["messages"]

                    # 1. Try applying to System Message
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system" and isinstance(
                            msg.get("content"), list
                        ):
                            longest_index, max_len = -1, -1
                            for j, part in enumerate(msg["content"]):
                                if part.get("type") == "text":
                                    text_len = len(part.get("text", ""))
                                    if text_len > max_len:
                                        max_len, longest_index = text_len, j
                            if longest_index != -1:
                                msg["content"][longest_index]["cache_control"] = {
                                    "type": "ephemeral"
                                }
                                cache_applied = True
                                break

                    # 2. Fallback to Last User Message
                    if not cache_applied:
                        for msg in reversed(messages):
                            if msg.get("role") == "user" and isinstance(
                                msg.get("content"), list
                            ):
                                longest_index, max_len = -1, -1
                                for j, part in enumerate(msg["content"]):
                                    if part.get("type") == "text":
                                        text_len = len(part.get("text", ""))
                                        if text_len > max_len:
                                            max_len, longest_index = text_len, j
                                if longest_index != -1:
                                    msg["content"][longest_index]["cache_control"] = {
                                        "type": "ephemeral"
                                    }
                                    break
                except Exception as cache_err:
                    print(f"Warning: Error applying cache_control logic: {cache_err}")
                    traceback.print_exc()
            # --- End Cache Control Logic ---

            # --- Apply Reasoning Logic ---
            if self.valves.INCLUDE_REASONING:
                # Validate reasoning effort level
                valid_efforts = ["high", "medium", "low"]
                if effort_level not in valid_efforts:
                    print(
                        f"Warning: Invalid reasoning effort '{effort_level}', defaulting to 'medium'"
                    )
                    effort_level = "medium"

                # Add reasoning configuration to payload
                payload["reasoning"] = {"effort": effort_level}
            # --- End Reasoning Logic ---

            # --- Apply Usage Tracking ---
            if self.valves.SHOW_USAGE_STATS:
                payload["usage"] = {"include": True}
            # --- End Usage Tracking ---

            # --- Apply Model-Specific Provider Blacklist ---
            blacklist_notification = ""
            if self.valves.MODEL_PROVIDER_BLACKLIST and "model" in payload:
                model_id = payload["model"]
                blacklist_str = self.valves.MODEL_PROVIDER_BLACKLIST.strip()
                ignored_providers = []
                
                try:
                    # Parse format: 'model1:provider1,provider2;model2:provider3'
                    for model_rule in blacklist_str.split(';'):
                        if ':' in model_rule:
                            model_pattern, providers_str = model_rule.split(':', 1)
                            model_pattern = model_pattern.strip().lower()
                            
                            # Check if current model matches the pattern
                            model_name_part = (
                                model_id.split("/", 1)[1].lower()
                                if "/" in model_id
                                else model_id.lower()
                            )
                            
                            # Support partial matching for model names
                            if model_pattern in model_name_part or model_pattern in model_id.lower():
                                providers = [
                                    p.strip().lower() 
                                    for p in providers_str.split(',') 
                                    if p.strip()
                                ]
                                ignored_providers.extend(providers)
                
                    # Apply provider blacklist to request if any providers should be ignored
                    if ignored_providers:
                        # Remove duplicates while preserving order
                        ignored_providers = list(dict.fromkeys(ignored_providers))
                        
                        # Add provider object to payload
                        if "provider" not in payload:
                            payload["provider"] = {}
                        payload["provider"]["ignore"] = ignored_providers
                        
                        # Create notification
                        blacklist_notification = f"> *Ignoring providers for {model_id}: {', '.join(ignored_providers)}*\n\n"
                        
                except Exception as e:
                    print(f"Warning: Error applying MODEL_PROVIDER_BLACKLIST: {e}")
            
            # Store blacklist notification for response formatting
            self._blacklist_notification = blacklist_notification
            # --- End Model-Specific Provider Blacklist ---

            headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": body.get("http_referer", "https://openwebui.com/"),
                "X-Title": body.get("x_title", "Open WebUI via Pipe"),
            }
            url = "https://openrouter.ai/api/v1/chat/completions"
            is_streaming = body.get("stream", False)

            # Ensure notifications are not included in model context
            if "messages" in payload:
                payload["messages"] = _sanitize_messages_for_notifications(payload["messages"])  # noqa: E501

            if is_streaming:
                return self.stream_response(
                    url,
                    headers,
                    payload,
                    _insert_citations,
                    _format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )
            else:
                content, usage = self.non_stream_response(
                    url,
                    headers,
                    payload,
                    _insert_citations,
                    _format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )
                return {"content": content, "usage": usage}

        except Exception as e:
            print(f"Error preparing request in pipe method: {e}")
            traceback.print_exc()
            return f"Pipe Error: Failed to prepare request: {e}"

    def non_stream_response(
        self, url, headers, payload, citation_inserter, citation_formatter, timeout
    ) -> str:
        """Handles non-streaming API requests."""
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            res = response.json()
            if not res.get("choices"):
                return ""

            choice = res["choices"][0]
            message = choice.get("message", {})
            citations = res.get("citations", [])
            usage = res.get("usage", {})

            content = message.get("content", "")
            reasoning = message.get("reasoning", "")

            content = citation_inserter(content, citations)
            reasoning = citation_inserter(reasoning, citations)
            citation_list = citation_formatter(citations)

            final = ""

            # Add effort notification if present
            effort_notification = getattr(self, "_effort_notification", "")
            if effort_notification:
                final += effort_notification

            # Add blacklist notification if present
            blacklist_notification = getattr(self, "_blacklist_notification", "")
            if blacklist_notification:
                final += blacklist_notification

            if reasoning:
                final += f"<think>\n{reasoning}\n</think>\n\n"
            if content:
                final += content
            if final:
                final += citation_list
            return final, usage

        except requests.exceptions.Timeout:
            return f"Pipe Error: Request timed out ({timeout}s)"
        except requests.exceptions.HTTPError as e:
            error_msg = f"Pipe Error: API returned HTTP {e.response.status_code}"
            try:
                detail = e.response.json().get("error", {}).get("message", "")
                if detail:
                    error_msg += f": {detail}"
            except Exception:
                pass
            return error_msg
        except Exception as e:
            print(f"Unexpected error in non_stream_response: {e}")
            traceback.print_exc()
            return f"Pipe Error: Unexpected error processing response: {e}"

    def stream_response(
        self, url, headers, payload, citation_inserter, citation_formatter, timeout
    ) -> Generator[str, None, None]:
        """Handles streaming API requests using a generator."""
        response = None
        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            )
            response.raise_for_status()

            in_think = False
            latest_citations: List[str] = []
            latest_usage = {}
            first_chunk = True

            for line in response.iter_lines(chunk_size=1024):
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[len(b"data: ") :].decode("utf-8")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if "choices" in chunk:
                    choice = chunk["choices"][0]
                    citations = chunk.get("citations")
                    if citations is not None:
                        latest_citations = citations
                    
                    # Track usage information
                    usage = chunk.get("usage")
                    if usage is not None:
                        latest_usage = usage
                    
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning", "")

                    # Add effort notification at the beginning
                    if first_chunk:
                        effort_notification = getattr(self, "_effort_notification", "")
                        if effort_notification:
                            yield effort_notification
                        
                        # Add blacklist notification if present
                        blacklist_notification = getattr(self, "_blacklist_notification", "")
                        if blacklist_notification:
                            yield blacklist_notification
                        
                        first_chunk = False

                    # Stream reasoning immediately
                    if reasoning:
                        if not in_think:
                            yield "<think>\n"
                            in_think = True
                        yield citation_inserter(reasoning, latest_citations)

                    # Stream content immediately
                    if content:
                        if in_think:
                            yield "\n</think>\n\n"
                            in_think = False
                        yield citation_inserter(content, latest_citations)

            # If model ended while still in <think>, close it
            if in_think:
                yield "\n</think>\n\n"
            
            # Append citations list
            yield citation_formatter(latest_citations)

            # Yield final usage data if available
            if latest_usage:
                yield f"data: {json.dumps({'usage': latest_usage})}\n\n"

        except GeneratorExit:
            if response:
                response.close()
            return
        except requests.exceptions.Timeout:
            yield f"Pipe Error: Request timed out ({timeout}s)"
        except requests.exceptions.HTTPError as e:
            yield f"Pipe Error: API returned HTTP {e.response.status_code}"
        except Exception as e:
            yield f"Pipe Error: Unexpected error during streaming: {e}"
        finally:
            if response:
                response.close()
