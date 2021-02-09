import numpy as np
import os


DATA_ROOT='misc/'

fs = 48000 # 采样频率
if fs==48000:
    alpha=0.58
    acoustic_n_out = 199
elif fs==22050:
    alpha=0.476
    acoustic_n_out = 190
else:
    alpha=0.58
frame_period = 5
hop_length = 80
fftlen =  1024
#alpha = 0.58

#声明模型输入输入维度
acoustic_n_in=471
#acoustic_n_out=199 # fs=16000 ,acoustic_n_out=187

#时长模型输入输出维度
duration_n_in = 467
duration_n_out = 1


# 声学特征下标分界点
mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

#三角窗系数
windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]


## 问题集加载路径
hed_path=os.path.join(DATA_ROOT, "questions-mandarin.hed")

#声学模型输入标准化处理参数 (0.01,0.99)
acoustic_mms_path = os.path.join(DATA_ROOT , 'X_acoustic_mms.pkl')
#声学模型输出标准化处理参数 均值-方差
acoustic_std_path = os.path.join(DATA_ROOT , 'Y_acoustic_std.pkl')

#时长模型输入标准化处理参数 (0.01,0.99)
duration_mms_path = os.path.join(DATA_ROOT , 'X_duration_mms.pkl')
#时长模型输出标准化处理参数 均值-方差
duration_std_path = os.path.join(DATA_ROOT ,'Y_duration_std.pkl')

# 时长模型和声学模型加载路径
duration_model_path = os.path.join(DATA_ROOT , 'duration_DNN.h5')
acoustic_model_path = os.path.join(DATA_ROOT , 'acoustic_GRU.5')


