import wfdb
import pandas as pd
import numpy as np
import os


def convert_to_csv(dat_path, output_path, channels=None):
    """
    把 GRABMyo .dat/.hea 文件转成项目用的 CSV 格式
    channels: 选择哪些通道，None则选前3个
    """
    record = wfdb.rdrecord(dat_path)

    print(f"采样率: {record.fs} Hz")
    print(f"通道数: {record.n_sig}")
    print(f"通道名: {record.sig_name}")
    print(f"时长: {record.sig_len / record.fs:.2f} 秒")

    # 选取通道，默认用 W1 W2 W3（腕部肌肉，和可穿戴设备最相关）
    if channels is None:
        sig_names = record.sig_name
        # 优先选腕部通道
        wrist_channels = [i for i, n in enumerate(sig_names) if n.startswith('W')]
        selected = wrist_channels[:3] if len(wrist_channels) >= 3 else [0, 1, 2]
    else:
        selected = channels

    selected_names = [record.sig_name[i] for i in selected]
    print(f"选取通道: {selected_names}")

    # 提取信号
    signals = record.p_signal[:, selected]

    # 转成 DataFrame
    df = pd.DataFrame(signals, columns=['channel1', 'channel2', 'channel3'])

    # 保存
    df.to_csv(output_path, index=False)
    print(f"已保存到: {output_path}")
    return df


# 转换单个文件
base_path = r"D:\PyCharm\opt\emgagent\data\grabmyo\session1_participant1_gesture1_trial1"
output_csv = r"D:\PyCharm\opt\emgagent\data\sample_emg.csv"

df = convert_to_csv(base_path, output_csv)
print(f"\n数据预览：\n{df.head()}")
print(f"数据形状：{df.shape}")

# 同时生成多个 trial 的数据用于对比实验
print("\n生成多 trial 对比数据...")
for trial in range(1, 8):
    trial_path = f"D:\\PyCharm\\opt\\emgagent\\data\\grabmyo\\session1_participant1_gesture1_trial{trial}"
    output = f"D:\\PyCharm\\opt\\emgagent\\data\\trial{trial}_emg.csv"
    if os.path.exists(trial_path + ".hea"):
        convert_to_csv(trial_path, output)
        print(f"trial{trial} 转换完成")
    else:
        print(f"trial{trial} 文件不存在，跳过")