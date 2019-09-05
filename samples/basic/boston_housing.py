from keras.datasets import boston_housing
from keras import *
import numpy as np

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]],dtype='float64')
mean = a.mean(axis=0)
print(mean)
a -= mean
print(a)
exit()

'''
train_data有404个样本，每个样本都有13个数值特征，比如人均犯罪率、每个住宅的平均房间数、高速公路可达性等
train_targets为房屋价格的中位数
'''
(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()

'''
将取值范围差异很大的数据输入到神经网络中，网络可能会自动适应这种取值范围不同的数据，学习变得更加困难。
解决方案：数据标准化，对每个特征做标准化，即对输入数据的每个特征减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为1 特征的取值都应该在相同的范围内
'''
mean = train_data.mean(axis=0)  #axis=0对各列求均值 axis=1对各行求均值 axis不设值对整个矩阵求均值
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

#在工作流程中不能使用在测试数据上计算得到的任何结果
test_data -= mean
test_data /= std

'''
构建网络
一般来说，训练数据越少，过拟合会越严重，而较小的网络可以降低过拟合
'''
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1)) #最后一层为线性层，添加激活函数会限制输出范围
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae']) #mse(mean squared error):均方误差 预测值与目标值之差的平方 mae(mean absolute error):平均绝对误差 预测值与目标值之差的绝对值
    return model

'''
目的：在少量数据条件下评估模型性能，获取最佳参数
由于数据集很少，验证集会非常小，因此验证分数可能有很大波动
解决方案：用K折验证来验证方法 
将可用数据划分为K个分区（K通常取4或5），实例化K个相同的模型，将每个模型在K-1个分区上训练，并在剩下的一个分区上进行评估。模型的验证分数等于K个验证分数的平均值

'''
k = 4
num_val_samples = len(train_data)//k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    model = build_model()
    model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=1)
    val_mse, val_mae = model.evaluate(val_data,val_targets,verbose=1)
    all_scores.append(val_mae)

print(all_scores)

