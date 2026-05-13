class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, func, description: str):
        self.tools[name] = {"func": func, "description": description}

    def get_tool_descriptions(self) -> str:
        desc = []
        for name, info in self.tools.items():
            desc.append(f"- {name}: {info['description']}")
        return "\n".join(desc)

    def call(self, name: str, **kwargs):
        if name not in self.tools:
            return f"错误：工具 {name} 不存在"
        return self.tools[name]["func"](**kwargs)