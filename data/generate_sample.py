import numpy as np
import pandas as pd

fs = 1000
t = np.linspace(0, 10, fs * 10)

# 模拟三通道肌电信号
np.random.seed(42)
ch1 = np.random.randn(len(t)) * 0.5 + np.sin(2 * np.pi * 30 * t) * 0.3
ch2 = np.random.randn(len(t)) * 0.3 + np.sin(2 * np.pi * 60 * t) * 0.2
ch3 = np.random.randn(len(t)) * 0.8 + np.sin(2 * np.pi * 20 * t) * 0.4

df = pd.DataFrame({'channel1': ch1, 'channel2': ch2, 'channel3': ch3})
df.to_csv('sample_emg.csv', index=False)
print("测试数据生成完成：sample_emg.csv")