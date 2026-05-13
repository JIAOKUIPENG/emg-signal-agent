import numpy as np
import pandas as pd
from scipy import signal

def load_emg(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        n_channels = len(df.columns)
        duration = len(df) / 2048  # 采样率2048Hz
        return f"已加载EMG数据：{n_channels}个通道，{len(df)}个采样点，时长约{duration:.1f}秒，列名：{list(df.columns)}"
    except Exception as e:
        return f"加载失败：{e}"

def preprocess_emg(file_path: str, channel: str) -> str:
    # 在 extract_features 函数开头加
    valid_channels = ['channel1', 'channel2', 'channel3']
    if channel not in valid_channels:
        channel = 'channel1'  # 默认回退
    try:
        df = pd.read_csv(file_path)
        raw = df[channel].values
        fs = 2048  # 采样率2048Hz

        # 带通滤波 20-450Hz
        b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)
        filtered = signal.filtfilt(b, a, raw)

        # 陷波滤波去除50Hz工频干扰
        b_notch, a_notch = signal.iirnotch(50, 30, fs)
        cleaned = signal.filtfilt(b_notch, a_notch, filtered)

        return (f"通道{channel}预处理完成："
                f"原始幅值范围[{raw.min():.3f}, {raw.max():.3f}]mV，"
                f"滤波后范围[{cleaned.min():.3f}, {cleaned.max():.3f}]mV，"
                f"已去除工频干扰和基线漂移")
    except Exception as e:
        return f"预处理失败：{e}"