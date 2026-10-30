"""Process-local compatibility for LiteLLM 1.93.0's Anthropic stream adapter.

The adapter can emit one block type per delta, so separate reasoning from
simultaneous answer content before its existing state machine consumes them.
Does not change model output, finish reasons, retries, or Gateway behavior.
"""
from copy import deepcopy


def normalize_reasoning_history(messages):
    """Preserve Anthropic thinking as OpenAI reasoning, with an explicit empty body."""
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            normalized.append(message)
            continue
        blocks = message.get('thinking_blocks') or []
        reasoning = ''.join(block['thinking'] for block in blocks
                            if isinstance(block, dict) and block.get('type') == 'thinking'
                            and isinstance(block.get('thinking'), str))
        additions = {}
        if reasoning and 'reasoning_content' not in message:
            additions['reasoning_content'] = reasoning
        if (blocks or message.get('reasoning_content')) and 'content' not in message and 'tool_calls' not in message:
            additions['content'] = ''
        normalized.append({**message, **additions} if additions else message)
    return normalized


def install():
    import litellm
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler as Handler,
    )
    from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
        _CombinedChunkSplitter,
    )

    original = _CombinedChunkSplitter._split
    if getattr(original, '_fusion_mixed_delta', False):
        return

    route_original = Handler._route_openai_thinking_to_responses_api_if_needed

    def respect_chat_route(completion_kwargs, *, thinking):
        # Honor the explicit Messages->Chat choice for this Chat-compatible
        # backend; do not silently undo it merely because thinking is enabled.
        if litellm.use_chat_completions_url_for_anthropic_messages:
            return
        return route_original(completion_kwargs, thinking=thinking)

    Handler._route_openai_thinking_to_responses_api_if_needed = staticmethod(respect_chat_route)

    def split(chunk):
        result = []
        for part in original(chunk):
            choices = getattr(part, 'choices', None)
            delta = getattr(choices[0], 'delta', None) if choices else None
            if delta is None:
                result.append(part)
                continue
            reason = bool(getattr(delta, 'reasoning_content', None) or getattr(delta, 'thinking_blocks', None))
            text = bool(getattr(delta, 'content', None))
            calls = getattr(delta, 'tool_calls', None) or []
            if int(reason) + int(text) + len(calls) <= 1:
                result.append(part)
                continue
            channels = ([('reasoning', None)] if reason else []) + ([('text', None)] if text else [])
            channels += [('tool', call) for call in calls]
            for channel, call in channels:
                item = deepcopy(part)
                piece = item.choices[0].delta
                if channel != 'text':
                    piece.content = None
                piece.tool_calls = [call] if channel == 'tool' else None
                if channel != 'reasoning':
                    for field in ('reasoning_content', 'thinking_blocks'):
                        if hasattr(piece, field):
                            delattr(piece, field)
                result.append(item)
        return result

    split._fusion_mixed_delta = True
    _CombinedChunkSplitter._split = staticmethod(split)
