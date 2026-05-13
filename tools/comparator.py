import pandas as pd
import numpy as np
from scipy import signal


def compare_trials(file1: str, file2: str) -> str:
    """对比两个trial的EMG特征差异"""
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        fs = 2048

        results = []
        for ch in ['channel1', 'channel2', 'channel3']:
            # 滤波
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)

            sig1 = signal.filtfilt(b, a, df1[ch].values)
            sig2 = signal.filtfilt(b, a, df2[ch].values)

            # 特征计算
            rms1 = np.sqrt(np.mean(sig1 ** 2))
            rms2 = np.sqrt(np.mean(sig2 ** 2))

            freqs1, psd1 = signal.welch(sig1, fs=fs)
            freqs2, psd2 = signal.welch(sig2, fs=fs)

            mpf1 = np.sum(freqs1 * psd1) / np.sum(psd1)
            mpf2 = np.sum(freqs2 * psd2) / np.sum(psd2)

            # 变化率
            rms_change = (rms2 - rms1) / rms1 * 100
            mpf_change = (mpf2 - mpf1) / mpf1 * 100

            # 疲劳判断
            if mpf_change < -2:
                fatigue = f"疲劳加深（MPF下降{abs(mpf_change):.1f}%）"
            elif mpf_change > 2:
                fatigue = f"激活增强（MPF上升{mpf_change:.1f}%）"
            else:
                fatigue = "状态稳定"

            results.append(
                f"{ch}:\n"
                f"  RMS: {rms1:.4f}→{rms2:.4f}mV（变化{rms_change:+.1f}%）\n"
                f"  MPF: {mpf1:.1f}→{mpf2:.1f}Hz（变化{mpf_change:+.1f}%）\n"
                f"  评估: {fatigue}"
            )

        return "Trial对比分析结果：\n" + "\n".join(results)

    except Exception as e:
        return f"对比分析失败：{e}"