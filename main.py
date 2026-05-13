import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.core import EMGAgent
from tools.signal_processor import load_emg, preprocess_emg
from tools.feature_extractor import extract_features
from tools.rag_retriever import search_knowledge
from tools.report_generator import generate_report

EMG_FILE = "data/sample_emg.csv"

def tool_load_emg() -> str:
    return load_emg(EMG_FILE)

def tool_preprocess(channel: str) -> str:
    return preprocess_emg(EMG_FILE, channel)

def tool_extract(channel: str) -> str:
    return extract_features(EMG_FILE, channel)


agent = EMGAgent()
agent.register_tool("load_emg", tool_load_emg,
    "加载EMG数据，无需参数：load_emg()")
agent.register_tool("preprocess_emg", tool_preprocess,
    "预处理指定通道，参数：channel（channel1/channel2/channel3）")
agent.register_tool("extract_features", tool_extract,
    "提取指定通道特征，参数：channel（channel1/channel2/channel3）")
agent.register_tool("search_knowledge", search_knowledge,
    "检索EMG专业知识，参数：query（如'MPF疲劳'/'凝胶电极阻抗'）")
agent.register_tool("generate_report", generate_report,
    "生成可视化报告，参数：summary（分析结论文字）")

result = agent.run_fast(
    "分析肌电信号，严格按以下顺序每次只执行一个Action：\n"
    "1. load_emg() 加载数据\n"
    "2. preprocess_emg channel1 → extract_features channel1\n"
    "3. preprocess_emg channel2 → extract_features channel2\n"
    "4. preprocess_emg channel3 → extract_features channel3\n"
    "5. search_knowledge query='MPF疲劳评估'\n"
    "6. generate_report 传入完整的中文疲劳评估结论\n"
    "7. Finish 输出最终总结"
)
print(f"\n最终结论：\n{result}")