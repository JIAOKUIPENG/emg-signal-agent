import json
import os
from datetime import datetime

class Memory:
    def __init__(self, path="memory/history.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.records = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save(self, user_input: str, result: str, structured: dict = None):
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "result": result[:200],  # 只保存前200字，节省空间
        }
        if structured:
            record["data"] = structured
        self.records.append(record)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def get_recent(self, n=3) -> str:
        if not self.records:
            return ""
        recent = self.records[-n:]
        return "\n".join([
            f"[{r['time']}] 用户：{r['user'][:50]}...\n结论：{r['result'][:100]}..."
            for r in recent
        ])