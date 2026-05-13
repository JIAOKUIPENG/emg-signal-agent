import time
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL_ID")

    def chat(self, messages: list, max_tokens: int = 500) -> str:
        """普通对话，用于 Reflection 自检"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM调用失败（第{attempt + 1}次）：{e}")
                if attempt < 2:
                    print("等待15秒后重试...")
                    time.sleep(15)
        return "错误：LLM调用失败"

    def chat_with_tools(self, messages: list, tools: list):
        """原生 Function Calling"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                return response.choices[0].message
            except Exception as e:
                print(f"Function Calling 失败（第{attempt+1}次）：{e}")
                if attempt < 2:
                    print("等待15秒后重试...")
                    time.sleep(15)
        return None

    def chat_stream(self, messages: list):
        """流式输出"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                return
            except Exception as e:
                print(f"流式调用失败（第{attempt+1}次）：{e}")
                if attempt < 2:
                    time.sleep(15)