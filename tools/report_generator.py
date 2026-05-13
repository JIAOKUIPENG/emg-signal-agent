import matplotlib
matplotlib.use('Agg')  # 非交互模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy import signal
from datetime import datetime
import os

def generate_report(summary: str) -> str:
    """生成包含波形图和频谱图的可视化报告"""
    try:
        # 读取数据
        df = pd.read_csv("data/sample_emg.csv")
        fs = 2048
        t = np.linspace(0, len(df)/fs, len(df))
        channels = ['channel1', 'channel2', 'channel3']

        # 创建画布，3行2列
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('EMG Signal Analysis Report', fontsize=14, fontweight='bold')

        colors = ['#2196F3', '#4CAF50', '#FF5722']

        for i, ch in enumerate(channels):
            raw = df[ch].values

            # 滤波
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)
            filtered = signal.filtfilt(b, a, raw)

            # 左列：时域波形
            axes[i, 0].plot(t[:2000], filtered[:2000], color=colors[i], linewidth=0.8)
            axes[i, 0].set_title(f'{ch} - Waveform')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude (mV)')
            axes[i, 0].grid(True, alpha=0.3)

            # 右列：频谱图
            freqs, psd = signal.welch(filtered, fs=fs)
            axes[i, 1].plot(freqs, psd, color=colors[i], linewidth=0.8)
            axes[i, 1].set_xlim(0, 500)
            axes[i, 1].set_title(f'{ch} - Power Spectrum')
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('PSD (mV²/Hz)')
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"reports/emg_report_{timestamp}.png"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 生成 Markdown 报告
        report_path = f"reports/emg_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# EMG Signal Analysis Report\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 分析结论\n\n{summary}\n\n")
            f.write(f"## 可视化图表\n\n")
            f.write(f"![EMG Report]({os.path.basename(img_path)})\n\n")
            f.write(f"## 信号参数\n\n")
            f.write(f"- 采样率：{fs} Hz\n")
            f.write(f"- 通道数：{len(channels)}\n")
            f.write(f"- 采样点数：{len(df)}\n")
            f.write(f"- 信号时长：{len(df)/fs:.1f} 秒\n")

        return f"报告生成成功：{report_path}，图表：{img_path}"

    except Exception as e:
        return f"报告生成失败：{e}"