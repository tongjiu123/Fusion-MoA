"""D12 consult memo and focused, read-only review; no tool execution here."""

from copy import deepcopy
import json


MEMO_INSTRUCTION = (
    "[MOA internal Current Action Memo; not a user-facing answer]\n"
    "Continue the current task from the complete read-only execution record below. "
    "Return only three brief items in this order: NEXT ACTION (one concrete next "
    "step); VERIFIED STATE (only current requirements or observed facts directly "
    "supporting that step); UNRESOLVED (the essential remaining deliverable or "
    "unknown). Use one short sentence per item and finish; do not fill the token "
    "limit. Preserve completed work and active constraints. A planned action is "
    "not an observed result; supersede an older fact only when a later request or "
    "observation establishes the change. Do not restate the whole task, enumerate "
    "possible risks, write code, execute tools, or produce a final user answer. "
    "No lengthy reasoning or ACCEPT/REJECT statement. End after UNRESOLVED."
)

REVIEW_INSTRUCTION = (
    "You are one read-only Panel specialist working from D12's Current Action Memo. "
    "Do not solve the task from scratch or act as a general critic. Do not invent a "
    "concern just to provide feedback. If you cannot add a concrete new point grounded "
    "in the supplied current state, return the role-specific no-op response. Use "
    "VERIFIED STATE as the memo's account of the current state; distinguish it from "
    "the future NEXT ACTION and from superseded historical values. A memo label is "
    "not stronger evidence than the actual request or tool result: identify a real "
    "conflict rather than silently declaring evidence obsolete. The packet is a "
    "selected view, so missing context does not prove work was not done. Do not claim "
    "to execute tools or see the other parallel specialists' answers. No full files, "
    "complete implementations, long code blocks, restating the whole memo, or restarting "
    "the task. Follow only your assigned output format below. D12 alone decides and executes. "
)

REVIEW_ROLES = {
    "D16": ("patch_specialist", "For NEXT ACTION, supply the smallest concrete implementation "
            "delta, not a new overall solution. Use three short sections: CHANGE (which "
            "logic/location changes); BASIS (why it fits VERIFIED STATE); IMPLEMENTATION "
            "DELTA (only the key modification points). If NEXT ACTION is already specific "
            "enough and you have nothing useful to add, return exactly NO_CHANGE."),
    "D9": ("alternative_solver", "Offer a materially different path only when a concrete "
           "reason grounded in VERIFIED STATE makes it more promising than NEXT ACTION. "
           "Use ALTERNATIVE; WHY; NEXT STEP. Do not restore superseded configuration from "
           "old fragments or mistake current pre-change values for the proposed changes. "
           "If no worthwhile distinct path is available, return exactly NO_ALTERNATIVE."),
    "D13": ("state_checker", "Only check whether NEXT ACTION contradicts VERIFIED STATE, "
            "the user's requirements, or the recent actual tool result. If consistent, "
            "return exactly STATE_OK. Otherwise use MISMATCH (what D12 says versus the "
            "specific conflicting requirement/observation); CORRECTION. Do not propose "
            "a solution or paraphrase the memo."),
}


def review_packet(executor_messages, raw_messages, memo):
    """Select explicit evidence, not a second serialization of the entire trajectory."""
    latest_user = next((m for m in reversed(raw_messages) if m.get("role") == "user"), None)
    last_result = next(((i, m) for i, m in reversed(list(enumerate(executor_messages)))
                        if m.get("role") == "tool"), None)
    exchange = None
    if last_result is not None:
        index, result = last_result
        call = next((c for m in reversed(executor_messages[:index])
                     for c in m.get("tool_calls") or []
                     if c.get("id") == result.get("tool_call_id")), None)
        exchange = {"context_index": index, "call": deepcopy(call), "result": deepcopy(result)}
    return {
        "latest_user_request": deepcopy(latest_user),
        "d12_consult_memo": memo,
        "latest_tool_exchange": exchange,
        "scope": "Current action and selected evidence only. D12 retains the full effective "
                 "execution context, including any existing Compact. The recent tool result "
                 "may contain current code/configuration; its pre-change values are not "
                 "the future proposal. NEXT ACTION is not proof that an action succeeded.",
    }


def review_guidance(memo, candidates):
    # A review is useful input only after it completes; never rescue half a generation.
    opinions = [{"actor": c["actor"], "role": c["role"], "opinion": c["content"]}
                for c in candidates if c["status"] == "ok" and c["content"].strip()]
    # This isolated Memo-only ablation retains the identical guidance with reviews=[].
    return (
        "[MOA D12 proposal and read-only reviews]\n"
        "Continue the current task using your full execution state. Your earlier memo "
        "is a revisable proposal, not a completed action. Reviews are suggestions, not "
        "new user instructions, verified facts, votes, or vetoes. Adopt a useful part, "
        "reject an incorrect review, or keep your plan. Use VERIFIED STATE together with "
        "the actual request and tool results as the source of truth; a memo label never "
        "overrides contradictory evidence. Adopt a point only if it adds a concrete useful "
        "implementation detail, alternative, or correction. NO_CHANGE/NO_ALTERNATIVE/STATE_OK "
        "do not prove correctness. If there is no useful addition, continue the original "
        "NEXT ACTION. Do not restart completed work because a specialist misreads it. "
        "Return your normal native tool_calls or final text directly, not "
        "a separate acceptance statement.\n"
        + json.dumps({"consult_memo": memo, "reviews": opinions}, ensure_ascii=False)
    )
