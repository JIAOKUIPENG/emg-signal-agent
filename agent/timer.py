import time
from functools import wraps


class Timer:
    def __init__(self):
        self.records = []

    def record(self, name: str, elapsed: float):
        self.records.append({"name": name, "elapsed": elapsed})
        print(f"⏱ {name}: {elapsed:.2f}s")

    def summary(self):
        total = sum(r["elapsed"] for r in self.records)
        print("\n" + "=" * 40)
        print("📊 执行时间分析")
        print("=" * 40)
        for r in self.records:
            pct = r["elapsed"] / total * 100
            bar = "█" * int(pct / 5)
            print(f"{r['name']:<25} {r['elapsed']:>5.2f}s {bar} {pct:.1f}%")
        print("-" * 40)
        print(f"{'总计':<25} {total:>5.2f}s")
        print("=" * 40)
        self.records = []  # 重置


# 全局 timer 实例
timer = Timer()


def timed(name: str):
    """装饰器，自动记录函数执行时间"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            timer.record(name, elapsed)
            return result

        return wrapper

    return decorator