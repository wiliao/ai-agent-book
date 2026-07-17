"""
实验 9-2 演示：用 ReAct Agent + PineClaw Voice（模拟）完成一个电话任务。

运行：
    python demo.py

它会真实调用 OpenAI：一边驱动上层 ReAct Agent 决策，一边在 make_phone_call 内部
用 OpenAI 扮演被叫方（IVR + 客服）完成一整段多轮通话，最后打印：
  (a) Agent 的 ReAct 轨迹（思考 + 发起 make_phone_call）
  (b) 返回的结构化通话记录（多轮 transcript + 是否达成目标 + 关键字段）
  (c) Agent 基于通话结果向用户的最终汇报
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent import run_agent


def _hr(title: str = "") -> None:
    line = "─" * 72
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(line)


def _print_record(rec: dict) -> None:
    print(f"  call_id        : {rec['call_id']}")
    print(f"  被叫号码       : {rec['phone_number']}")
    print(f"  状态           : {rec['status']}  |  是否达成目标: {rec['goal_achieved']}")
    print(f"  通话时长(模拟) : {rec['duration_seconds']} 秒")
    print(f"  摘要           : {rec['summary']}")
    print("  关键字段(key_fields):")
    if rec["key_fields"]:
        for k, v in rec["key_fields"].items():
            print(f"      - {k}: {v}")
    else:
        print("      （无）")
    print(f"  需要追问       : {rec['follow_up_needed']}  {rec.get('follow_up_reason', '')}")
    print("  通话转录(transcript):")
    for turn in rec["transcript"]:
        speaker = turn["speaker"]
        # 简单对齐：语音Agent 用 >>，被叫方用 <<
        arrow = ">>" if speaker == "语音Agent" else "<<"
        print(f"      {arrow} [{speaker}] {turn['text']}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：未检测到 OPENAI_API_KEY。请复制 env.example 为 .env 并填入有效 key。")
        sys.exit(1)

    # 书中示例任务：注意这里只给了自然语言任务，Agent 需自行决定通话参数。
    task = (
        "帮我打电话给宽带客服（客服热线 10010），查询本月账单为什么多扣了 50 元，"
        "要求对方解释清楚原因，如果是误扣就请他们处理。我的宽带账号是 hz-88231。"
    )

    _hr("用户任务")
    print(task)

    _hr("ReAct Agent 轨迹")

    def on_event(kind: str, payload) -> None:
        if kind == "think":
            print(f"\n[Agent 思考] {payload}")
        elif kind == "call":
            print("\n[Agent 调用工具 make_phone_call] 入参:")
            print(f"    phone_number = {payload.get('phone_number')}")
            print(f"    goal         = {payload.get('goal')}")
            print(f"    context      = {payload.get('context', '')}")
        elif kind == "record":
            print("\n[PineClaw 返回结构化通话记录]")
            _print_record(payload)
        elif kind == "final":
            pass  # 最终汇报单独打印

    final = run_agent(task, on_event=on_event)

    _hr("Agent 向用户的最终汇报")
    print(final)
    print()


if __name__ == "__main__":
    main()
