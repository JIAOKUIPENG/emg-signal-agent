import gradio as gr
import sys, os
import time
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.core import EMGAgent
from agent.memory import Memory
from agent.llm import LLMClient
from agent.timer import timer
from tools.signal_processor import load_emg, preprocess_emg
from tools.feature_extractor import extract_features
from tools.rag_retriever import search_knowledge
from tools.report_generator import generate_report
from tools.comparator import compare_trials

EMG_FILE = "data/sample_emg.csv"
memory = Memory()

def build_agent():
    agent = EMGAgent()
    agent.register_tool("load_emg",
        lambda **kwargs: load_emg(EMG_FILE),
        "加载EMG数据，无需参数，直接调用：load_emg()")
    agent.register_tool("preprocess_emg",
        lambda channel, **kwargs: preprocess_emg(EMG_FILE, channel),
        "预处理通道，参数：channel（channel1/channel2/channel3）")
    agent.register_tool("extract_features",
        lambda channel, **kwargs: extract_features(EMG_FILE, channel),
        "提取特征，参数：channel（channel1/channel2/channel3）")
    agent.register_tool("search_knowledge",
        lambda query, **kwargs: search_knowledge(query),
        "检索知识库，参数：query")
    agent.register_tool("generate_report",
        lambda summary="EMG分析完成", **kwargs: generate_report(summary),
        "生成报告，参数：summary")

    def tool_compare(file1, file2, **kwargs):
        if not file1.endswith('.csv'):
            file1 = f"data/{file1}_emg.csv"
        if not file2.endswith('.csv'):
            file2 = f"data/{file2}_emg.csv"
        return compare_trials(file1, file2)

    agent.register_tool("compare_trials",
        tool_compare,
        "对比两个EMG文件特征差异，参数：file1和file2，可以是trial1/trial7简写或完整路径")

    return agent

def chat(message, history):
    agent = build_agent()
    timer.records = []  # 每次对话重置计时

    # 意图分类
    t = time.time()
    intent = agent.classify_intent(message)
    timer.record("意图分类", time.time() - t)

    # 根据意图决定是否带历史
    context_parts = []
    if intent == "FOLLOWUP" and history:
        context_lines = []
        for h in history[-3:]:
            if isinstance(h, dict):
                context_lines.append(str(h.get("content", "")))
            else:
                context_lines.append(f"用户：{h[0]}\n助手：{h[1]}")
        if context_lines:
            context_parts.append("本次对话历史：\n" + "\n".join(context_lines))
        past = memory.get_recent(3)
        if past:
            context_parts.append(f"历史分析记录：\n{past}")

    full_message = "\n\n".join(context_parts) + f"\n\n当前问题：{message}" if context_parts else message

    # 根据意图选执行方式
    if intent == "KNOWLEDGE":
        t2 = time.time()
        knowledge = search_knowledge(message)
        timer.record("RAG检索", time.time() - t2)

        t2 = time.time()
        llm = LLMClient()
        result = llm.chat([
            {"role": "system", "content":
                "你是肌电信号专家，根据知识库内容回答问题，300字以内，简洁专业。"},
            {"role": "user", "content": f"知识库：{knowledge}\n\n问题：{message}"}
        ])
        timer.record("LLM生成回答", time.time() - t2)

    elif intent == "COMPARE":
        t2 = time.time()
        result = agent.run_with_function_calling(full_message)
        timer.record("对比分析执行", time.time() - t2)

    else:
        # ANALYSIS / FOLLOWUP → run_fast
        t2 = time.time()
        plan_messages = [
            {"role": "system", "content": f"""你是肌电信号分析专家。
可用工具：
{agent.registry.get_tool_descriptions()}
请针对用户需求，一次性输出完整执行计划，严格按以下格式：
PLAN:
1. tool_name(arg="value")
只输出PLAN部分，不要其他内容。"""},
            {"role": "user", "content": full_message}
        ]
        plan_response = agent.llm.chat(plan_messages)
        timer.record("LLM规划", time.time() - t2)

        actions = re.findall(r'\d+\.\s*(\w+)\(([^)]*)\)', plan_response)
        results = []

        t2 = time.time()
        for tool_name, args_str in actions:
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str)) if args_str.strip() else {}
            try:
                obs = agent.registry.call(tool_name, **kwargs)
            except TypeError as e:
                obs = f"工具调用参数错误，跳过：{e}"
            results.append(f"[{tool_name}] {obs}")
        timer.record("工具执行（本地）", time.time() - t2)

        t2 = time.time()
        llm = LLMClient()
        result = llm.chat([
            {"role": "system", "content":
                "你是肌电信号分析专家，根据工具执行结果给出专业分析结论。"
                "要求：300字以内，包含关键数值和疲劳评估结论，语言简洁专业，不要分点列举。"},
            {"role": "user", "content":
                f"工具执行结果：\n" + "\n".join(results) + f"\n\n用户问题：{full_message}"}
        ])
        timer.record("LLM生成结论", time.time() - t2)

    # Memory写入
    t = time.time()
    memory.save(message, result)
    timer.record("Memory写入", time.time() - t)

    # 打印时间汇总
    timer.summary()

    # 流式返回
    output = ""
    for char in result:
        output += char
        yield output

demo = gr.ChatInterface(
    fn=chat,
    title="EMG Signal Analysis Agent",
    description="肌电信号智能分析系统，输入分析需求自动执行",
    examples=[
        "分析 data/sample_emg.csv 的三个通道，给出疲劳评估报告",
        "channel3 的疲劳状态如何？",
        "凝胶电极对信号质量有什么影响？",
    ]
)

demo.launch()