import re
from agent.llm import LLMClient
from agent.tool_registry import ToolRegistry

SYSTEM_PROMPT_TEMPLATE = """你是一个专业的肌电信号分析智能体。
你可以使用以下工具：
{tool_descriptions}

每次回复必须严格遵循格式：
Thought: 你的分析思路
Action: 工具名称(参数名="参数值")

当分析完成时使用：
Action: Finish[最终分析结论]

要求：
- 每次只输出一对Thought-Action，不要多余内容
- Finish时的结论要详细、专业，至少包含3-5句话
- 引用知识库内容时要结合实际场景解释
"""

class EMGAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.registry = ToolRegistry()

    def register_tool(self, name, func, description):
        self.registry.register(name, func, description)

    def run(self, user_input: str, max_steps: int = 15) -> str:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self.registry.get_tool_descriptions()
        )
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            # 超过10轮只保留系统prompt+最近6轮
            if len(history) > 10:
                history = history[:1] + history[-6:]
            response = self.llm.chat(history)
            print(f"Agent: {response}")
            history.append({"role": "assistant", "content": response})

            # 解析 Action
            action_match = re.search(r"Action:\s*(.*)", response)
            if not action_match:
                obs = "错误：未找到 Action，请严格按格式输出"
                history.append({"role": "user", "content": f"Observation: {obs}"})
                continue

            action_str = action_match.group(1).strip()

            # 判断是否结束
            if action_str.startswith("Finish"):
                final = re.search(r"Finish\[(.*)\]", action_str, re.DOTALL)
                conclusion = final.group(1) if final else "任务完成"

                # Reflection：让模型自检结论
                print("\n--- Reflection 自检 ---")
                reflection_messages = [
                    {"role": "system", "content": "你是一个严格的肌电信号分析质检专家。"},
                    {"role": "user", "content":
                        f"请检查以下EMG分析结论是否合理，重点检查：\n"
                        f"1. MPF/MDF数值是否与疲劳状态描述一致\n"
                        f"2. 三个通道是否都有分析\n"
                        f"3. 建议是否具体可执行\n\n"
                        f"结论：{conclusion}\n\n"
                        f"如果合理回复'通过'，如果有问题指出具体问题并给出修正后的结论。"
                     }
                ]
                reflection = self.llm.chat(reflection_messages)
                print(f"Reflection结果: {reflection}")

                if "通过" in reflection:
                    return conclusion
                else:
                    # 用修正后的结论
                    return reflection

            # 解析工具调用
            tool_match = re.match(r"(\w+)\((.*)\)", action_str, re.DOTALL)
            if not tool_match:
                obs = "错误：Action 格式不正确，应为 tool_name(arg='value')"
                history.append({"role": "user", "content": f"Observation: {obs}"})
                continue

            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str)) if args_str.strip() else {}

            # 调用工具
            print(f"调用工具: {tool_name}，参数: {kwargs}")
            obs = self.registry.call(tool_name, **kwargs)
            print(f"Observation: {obs}")
            collected = sum(1 for h in history if '特征提取完成' in h.get('content', ''))
            hint = "\n请注意：已收集足够数据，请立即使用 Action: Finish[结论] 输出最终报告。" if collected >= 3 else ""
            history.append({"role": "user", "content": f"Observation: {obs}{hint}"})

        return "达到最大步数，任务未完成"

    def run_fast(self, user_input: str) -> str:
        """快速模式：一次规划，顺序执行，最后综合结论"""

        # 第一次 LLM 调用：规划所有步骤
        plan_prompt = f"""你是肌电信号分析专家。
    可用工具：
    {self.registry.get_tool_descriptions()}

    请针对用户需求，一次性输出完整执行计划，严格按以下格式：
    PLAN:
    1. tool_name(arg="value")
    2. tool_name(arg="value")
    3. ...

    只输出PLAN部分，不要其他内容。
    """
        messages = [
            {"role": "system", "content": plan_prompt},
            {"role": "user", "content": user_input}
        ]

        plan_response = self.llm.chat(messages)
        print(f"\n执行计划：\n{plan_response}")

        # 解析并执行所有步骤
        import re
        actions = re.findall(r'\d+\.\s*(\w+)\(([^)]*)\)', plan_response)

        results = []
        for tool_name, args_str in actions:
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str)) if args_str.strip() else {}
            print(f"\n执行：{tool_name}({kwargs})")
            obs = self.registry.call(tool_name, **kwargs)
            print(f"结果：{obs}")
            results.append(f"[{tool_name}] {obs}")

        if not results:
            return "未能解析执行计划，请重试"

        # 第二次 LLM 调用：综合结论
        summary_messages = [
            {"role": "system", "content":
                "你是肌电信号分析专家，根据工具执行结果给出专业、详细的分析结论，最多5句话。"},
            {"role": "user", "content":
                f"工具执行结果：\n" + "\n".join(results) + f"\n\n用户问题：{user_input}"}
        ]

        print("\n生成最终结论...")
        return self.llm.chat(summary_messages)

    def run_with_function_calling(self, user_input: str) -> str:
        """原生 Function Calling + 并行执行工具"""
        import json
        import concurrent.futures

        # 构建工具定义
        tools = []
        for name, info in self.registry.tools.items():
            desc = info["description"]
            params = {}
            required = []

            if "channel" in desc:
                params["channel"] = {
                    "type": "string",
                    "description": "通道名称，可选值：channel1/channel2/channel3"
                }
                required.append("channel")
            if "query" in desc:
                params["query"] = {
                    "type": "string",
                    "description": "检索关键词"
                }
                required.append("query")
            if "summary" in desc:
                params["summary"] = {
                    "type": "string",
                    "description": "分析结论文字"
                }
            if "file1" in desc:
                params["file1"] = {"type": "string", "description": "第一个文件路径"}
                params["file2"] = {"type": "string", "description": "第二个文件路径"}
                required.extend(["file1", "file2"])

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required
                    }
                }
            })

        messages = [
            {"role": "system", "content":
                "你是专业的肌电信号分析专家，根据用户问题智能选择需要的工具。"
                "不是每次都要调用所有工具，根据问题按需调用。"
                "分析任务：加载→预处理→提取特征→检索→报告。"
                "知识问题：只用search_knowledge。"
                "对比任务：只用compare_trials。"},
            {"role": "user", "content": user_input}
        ]

        max_steps = 15
        called_tools = set()  # 记录已调用的工具

        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            message = self.llm.chat_with_tools(messages, tools)

            if message is None:
                return "Function Calling 调用失败"

            # 没有工具调用，直接返回结论
            if not message.tool_calls:
                print(f"最终结论生成完毕")
                return message.content or "分析完成"

            # 过滤重复工具调用
            new_calls = []
            for tool_call in message.tool_calls:
                tool_key = f"{tool_call.function.name}_{tool_call.function.arguments}"
                if tool_key not in called_tools:
                    called_tools.add(tool_key)
                    new_calls.append(tool_call)
                else:
                    print(f"跳过重复调用: {tool_call.function.name}")

            if not new_calls:
                # 所有工具都重复了，强制给结论
                messages.append({
                    "role": "user",
                    "content": "所有工具已执行完毕，请直接给出最终分析结论。"
                })
                return self.llm.chat(messages)

            # 解析所有工具调用
            pending = []
            for tool_call in new_calls:  # 注意这里用 new_calls
                tool_name = tool_call.function.name
                try:
                    kwargs = json.loads(tool_call.function.arguments)
                except:
                    kwargs = {}
                pending.append((tool_call, tool_name, kwargs))
                print(f"待执行: {tool_name}({kwargs})")

            # 并行执行所有工具
            print(f"并行执行 {len(pending)} 个工具...")
            tool_results = {}

            def execute_tool(item):
                tool_call, tool_name, kwargs = item
                try:
                    if isinstance(kwargs, str):
                        import json
                        kwargs = json.loads(kwargs)
                    obs = self.registry.call(tool_name, **kwargs)
                except Exception as e:
                    obs = f"工具执行失败：{e}"
                print(f"完成: {tool_name} → {str(obs)[:80]}...")
                return tool_call.id, obs

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(execute_tool, item) for item in pending]
                for future in concurrent.futures.as_completed(futures):
                    tool_call_id, obs = future.result()
                    tool_results[tool_call_id] = obs

            # 把结果加入对话
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })

            for tool_call in message.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_results.get(tool_call.id, "执行失败"))
                })

        return "达到最大步数，任务未完成"

    def classify_intent(self, message: str) -> str:  # ← 加在这里
            response = self.llm.chat([
                {"role": "system", "content":
                    "判断用户问题的类型，只回复以下之一：\n"
                    "ANALYSIS - 需要加载和分析EMG信号数据，如'分析三个通道'\n"
                    "COMPARE - 需要对比多个文件或trial，如'对比trial1和trial7'\n"
                    "KNOWLEDGE - 询问专业知识概念，如'MPF是什么'\n"
                    "FOLLOWUP - 追问上一次分析结果，如'为什么channel3最高'\n"
                    "只输出一个单词，不要解释。"
                 },
                {"role": "user", "content": message}
            ])
            intent = response.strip().upper()
            print(f"意图识别：{intent}")
            return intent

