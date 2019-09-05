import numpy as np 

'''简单RNN的前向传播 '''

timesteps = 3 #输入序列的时间步数
input_features = 8 #输入特征空间的维度
output_features = 16 #输出特征空间的维度

inputs = np.random.random((timesteps, input_features)) #输入数据：随机噪声，仅作为示例

state_t = np.zeros((output_features,)) #初始状态：全零向量

#创建随机的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W,input_t) + np.dot(U,state_t) + b) #由输入和当前状态（前一个输出）计算得到当前输出
    successive_outputs.append(output_t) #将这个输出保存到一个列表里
    state_t = output_t #更新网络状态，用于下一个时间步

print(output_t)
final_output_sequence = np.stack(successive_outputs, axis=0) #最终输出是一个形状为(timesteps,output_features)的二维向量
print(final_output_sequence)

