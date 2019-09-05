from keras.datasets import reuters
from keras.utils import *
import matplotlib.pyplot as plt
from keras import *
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#将数据向量化
#         字典索引0  1  2  3  4  5  6  7 ...
#sample 0:       1.    1.    1.
#sample 1：          1.    1.
#sample 2：                         1. 1.
#........
def vectorize_sequence(sequences, dimension=10000):
    result = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        result[i][sequence] = 1.
    return result

train_data = vectorize_sequence(train_data)
test_data = vectorize_sequence(test_data)

#训练标签向量化(one-hot编码)
#one-hot编码：用索引来表示类别
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

'''
#训练标签向量化(转换为整数张量)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
损失函数需变为 sparse_categorical_crossentropy
'''

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001),input_shape=(10000,))) #一次一篇评论
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#留出1000个样本作为验证集
val_data = train_data[:1000]
train_data = train_data[1000:]

val_labels = train_labels[:1000]
train_labels = train_labels[1000:]

#训练模型
history = model.fit(train_data,train_labels,epochs=50,batch_size=512,validation_data=(val_data,val_labels))


#绘制训练损失和验证损失
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()#显示图例
plt.show()

plt.clf()
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.legend()
plt.show()



predictions = model.predict(test_data)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))#第一个测试样本对应的类别是4




