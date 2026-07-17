"""
电话 Agent —— 一个把 PineClaw Voice（make_phone_call）当作工具的 ReAct Agent。

上层是标准的 ReAct 循环（OpenAI function calling 实现）：
    用户给一个任务（如"打电话给宽带客服，问清本月为何多扣 50 元并要求解释"）
        -> Agent 思考需要哪些参数（号码 / 目标 / 上下文）
        -> 调用 make_phone_call 工具完成整段通话
        -> 读取返回的【结构化通话记录】
        -> 若信息不足则追问/再拨，否则向用户给出最终汇报

工具 make_phone_call 由 pine_voice 模块提供（对真实 PineClaw Voice API 的本地模拟）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from openai import OpenAI

from pine_voice import make_phone_call

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)

# 最多允许 Agent 发起几次工具调用（含追问/再拨），防止死循环。
_MAX_STEPS = 6

_SYSTEM_PROMPT = (
    "你是一名『电话助理』Agent。用户会交给你一个需要打电话才能完成的任务，"
    "你要代表用户把电话打好并汇报结果。\n\n"
    "你有一个工具 make_phone_call，它会把整通电话（拨号、IVR 菜单导航、与客服多轮对话、"
    "转录）全部交给 PineClaw 语音 Agent 完成，并返回结构化通话记录。\n\n"
    "工作方式（ReAct）：\n"
    "1. 先想清楚要拨打的号码、通话目标、需要带上的上下文（账号/姓名/已知信息）。\n"
    "2. 调用 make_phone_call 完成通话。\n"
    "3. 读取返回的结构化记录（goal_achieved / key_fields / summary / transcript）。\n"
    "4. 若目标未达成或信息不足（follow_up_needed 为真），可以带着更明确的目标再拨一次"
    "（但总次数有限，不要无谓重拨）。\n"
    "5. 目标达成后，用简洁中文向用户汇报：结论 + 关键信息（金额/原因/确认号/时间等）"
    "+ 后续建议（如有）。\n\n"
    "注意：如果任务里连电话号码都没有，就用一个合理的占位号码并在汇报中说明。"
    "始终基于工具真实返回的通话记录来汇报，不要编造通话没有提到的信息。"
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_phone_call",
            "description": (
                "代表用户拨打一通真实电话。PineClaw 语音 Agent 会完成拨号、IVR 菜单导航、"
                "与对方多轮对话，并返回结构化通话记录（含 transcript、是否达成目标、"
                "抽取的关键字段）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "被叫电话号码，如 10000 或 400-810-xxxx。",
                    },
                    "goal": {
                        "type": "string",
                        "description": "本次通话要达成的明确目标（一句话）。",
                    },
                    "context": {
                        "type": "string",
                        "description": "辅助上下文：账号、姓名、已知金额、时间等，可为空。",
                    },
                },
                "required": ["phone_number", "goal"],
            },
        },
    }
]


def run_agent(task: str, on_event: Callable[[str, Any], None] | None = None) -> str:
    """
    运行 ReAct 电话 Agent。

    参数：
        task:     用户的自然语言任务。
        on_event: 可选回调，用于外部观察 Agent 轨迹。
                  事件类型：'think'(Agent 思考文本) / 'call'(工具入参) /
                  'record'(结构化通话记录) / 'final'(最终汇报)。

    返回：
        Agent 面向用户的最终汇报文本。
    """

    def emit(kind: str, payload: Any) -> None:
        if on_event:
            on_event(kind, payload)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for _ in range(_MAX_STEPS):
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = resp.choices[0].message

        # 附上模型这一步返回的 assistant 消息（可能同时含思考文本与工具调用）。
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            emit("think", msg.content)

        if not msg.tool_calls:
            # 没有工具调用 = Agent 给出了最终答复。
            final = msg.content or ""
            emit("final", final)
            return final

        # 逐个执行工具调用（本例只有 make_phone_call）。
        for tc in msg.tool_calls:
            if tc.function.name != "make_phone_call":
                result = {"error": f"未知工具 {tc.function.name}"}
            else:
                args = json.loads(tc.function.arguments or "{}")
                emit("call", args)
                result = make_phone_call(
                    phone_number=args.get("phone_number", "10000"),
                    goal=args.get("goal", task),
                    context=args.get("context", ""),
                )
                emit("record", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "make_phone_call",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # 兜底：步数用尽仍未收敛，逼模型直接总结一次。
    messages.append(
        {
            "role": "user",
            "content": "请根据以上通话记录，立即给用户一份最终汇报，不要再打电话了。",
        }
    )
    resp = _client.chat.completions.create(
        model=_MODEL, messages=messages, temperature=0.3
    )
    final = resp.choices[0].message.content or ""
    emit("final", final)
    return final
