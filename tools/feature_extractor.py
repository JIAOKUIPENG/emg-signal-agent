import numpy as np
import pandas as pd
from scipy import signal

def extract_features(file_path: str, channel: str) -> str:
    # 在 extract_features 函数开头加
    valid_channels = ['channel1', 'channel2', 'channel3']
    if channel not in valid_channels:
        channel = 'channel1'  # 默认回退
    try:
        df = pd.read_csv(file_path)
        raw = df[channel].values
        fs = 2048

        # 滤波
        b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)
        emg = signal.filtfilt(b, a, raw)

        # 时域特征
        rms = np.sqrt(np.mean(emg**2))
        mav = np.mean(np.abs(emg))
        zc = np.sum(np.diff(np.sign(emg)) != 0)  # 过零率

        # 频域特征
        freqs, psd = signal.welch(emg, fs=fs)
        total_power = np.trapezoid(psd, freqs)
        mpf = np.sum(freqs * psd) / np.sum(psd)  # 平均功率频率
        mdf = freqs[np.searchsorted(np.cumsum(psd), np.sum(psd)/2)]  # 中位频率


        # 疲劳指数（MPF下降反映疲劳）
        fatigue = "轻度疲劳" if mpf < 80 else "正常" if mpf < 120 else "高激活"

        return (f"通道{channel}特征提取完成：\n"
                f"  时域：RMS={rms:.3f}mV，MAV={mav:.3f}mV，过零率={zc}次\n"
                f"  频域：MPF={mpf:.1f}Hz，MDF={mdf:.1f}Hz\n"
                f"  总功率：{total_power:.4f}mV²/Hz\n"
                f"  疲劳状态：{fatigue}")
    except Exception as e:
        return f"特征提取失败：{e}"