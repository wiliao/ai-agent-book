# 实验 9-2：使用 PineClaw Voice API 构建电话 Agent

配套《深入理解 AI Agent》第 9 章实验 9-2。

## 目的

真实世界里很多 Agent 任务离不开**拨打真实电话**——联系客服协商账单、预约餐厅、
确认订单。本实验演示语音 Agent 的一个重要应用方向：**Agent 不仅能与用户语音对话，
还能代替用户与外部世界进行电话交互**。

上层是一个标准的 **ReAct Agent**：接到一个自然语言任务（如"打电话给宽带客服，
查询本月账单为何多扣了 50 元并要求解释"），它自己想清楚要拨的号码、通话目标和
上下文，调用 `make_phone_call` 工具完成整段通话，读取返回的**结构化通话记录**，
必要时追问/再拨，最后向用户汇报结果。

## 电话语音 API 的抽象

生产级电话语音 API（如 [PineClaw Voice API](https://pineclaw.com/)，作者团队开发）
把一整通电话封装成**一次工具调用**：

```
record = make_phone_call(phone_number, goal, context)
```

你只提供三样东西——**号码、目标、上下文**——它的语音 Agent 就会自动完成：

- **拨号**：接通被叫方；
- **IVR 导航**：应对"查询请按 1，转人工请按 0"这类按键菜单；
- **多轮对话**：接通人工后围绕目标交涉、追问、确认关键信息；
- **转录**：把整段通话转成文字。

最后返回一份**结构化通话记录**，而不是一段裸录音。这正是它能塞进 ReAct 循环的原因：
Agent 拿到的是结构化字段（是否达成目标、抽取的关键信息、逐轮 transcript），可以直接
据此决策与汇报。本实验返回体的形状（见 `pine_voice.py` 的 `CallRecord`）：

| 字段 | 含义 |
| --- | --- |
| `call_id` | 通话唯一 ID |
| `phone_number` / `goal` | 本次通话的号码与目标 |
| `status` / `goal_achieved` | 通话状态 / 是否达成目标 |
| `duration_seconds` | 通话时长 |
| `summary` | 一句话通话摘要 |
| `key_fields` | 抽取的关键信息（扣费原因 / 金额 / 确认号 / 时间…） |
| `transcript` | 逐轮对话 `[{speaker, text}, ...]` |
| `follow_up_needed` / `follow_up_reason` | 是否仍需追问及原因 |

## 关于 mock（重要）

真实 PineClaw Voice API 需要 `PINECLAW_API_KEY` 并会拨打真实电话号码。为便于离线跑通，
本实验**用一个本地模拟客户端替代真实 API**（`pine_voice.py`）：

- 它**不接触真实电话网络**，也**不需要 PineClaw key**；
- `make_phone_call` 内部用 **OpenAI 扮演被叫方**——先当自动 IVR 语音菜单，被"转人工"
  后再扮演人工客服——与去电的语音 Agent 进行一段**多轮对话**（模拟 IVR 导航 + 客服应答），
  然后把 transcript 归纳成上表的结构化记录；
- 关键在于：**模拟客户端与真实 API 的输入/输出契约完全一致**，因此上层 ReAct Agent
  的代码在切换到真实 PineClaw SDK 时几乎无需改动。

所以本实验里出现的"扣费原因""确认号"等都是**模型即时编造的模拟情节**，仅用于演示
数据流，不代表任何真实通话。

### 真实接入 PineClaw

把 `agent.py` 里对模拟 `make_phone_call` 的调用替换为真实 SDK 即可，其余逻辑不变：

```python
# pip install pine-voice
from pine_voice import PineVoiceClient   # 真实 SDK（示意）

client = PineVoiceClient(api_key=os.environ["PINECLAW_API_KEY"])

def make_phone_call(phone_number, goal, context=""):
    call = client.calls.create(to=phone_number, goal=goal, context=context)
    result = call.wait()          # 阻塞直到通话结束（可能是分钟级到小时级）
    return result.to_dict()       # 返回同形状的结构化通话记录
```

真实使用请以 PineClaw 官方文档为准；建议先拨打**自己的手机**验证连通性。

## 运行

```bash
cd chapter9/phone-agent
pip install -r requirements.txt

cp env.example .env
# 编辑 .env，填入有效的 OPENAI_API_KEY

python demo.py
```

`demo.py` 会真实调用 OpenAI，打印三段内容：
(a) ReAct Agent 的轨迹（思考 + 发起 `make_phone_call`）；
(b) 返回的结构化通话记录（多轮 transcript + 是否达成目标 + 关键字段）；
(c) Agent 基于通话结果向用户的最终汇报。

> 只使用 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL` 指向兼容网关，如 Moonshot / 火山方舟）。
> 请勿使用 OPENROUTER / ANTHROPIC / DEEPSEEK / SILICONFLOW。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `pine_voice.py` | PineClaw Voice API 的本地**模拟**客户端，提供 `make_phone_call` 工具 |
| `agent.py` | 把 `make_phone_call` 当工具的 **ReAct Agent**（OpenAI function calling） |
| `demo.py` | 端到端演示：一个电话任务从下达到汇报 |
| `requirements.txt` / `env.example` | 依赖与环境变量模板 |
