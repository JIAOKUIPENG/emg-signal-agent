import wfdb
import os

# 修改成你实际的文件路径（不需要带扩展名）
record_path = r"D:\PyCharm\opt\emgagent\data\grabmyo\session1_participant1_gesture1_trial1"

# 读取数据
record = wfdb.rdrecord(record_path)

print(f"采样率: {record.fs} Hz")
print(f"通道数: {record.n_sig}")
print(f"通道名: {record.sig_name}")
print(f"信号长度: {record.sig_len} 采样点")
print(f"时长: {record.sig_len / record.fs:.2f} 秒")
print(f"信号数据形状: {record.p_signal.shape}")
print(f"前5行数据:\n{record.p_signal[:5]}")