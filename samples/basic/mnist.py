from keras.datasets import mnist
from keras import models
from keras import layers 
from keras.utils import to_categorical

'''
获取数据
'''
(train_images, train_labels), (test_images, test_labels)= mnist.load_data()

'''
数据处理
将6000个二维矩阵变为6000个一维向量
数据归一化，0-1
标签编码，如：数字3的编码为[0,0,0,1,0,0,0,0,0,0]
'''
train_images = train_images.reshape(train_images.shape[0],-1)
test_images = test_images.reshape(test_images.shape[0],-1)

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)#数据处理
'''
print(test_images[0].shape)
print(test_labels[0])
exit()
'''

'''
构建网络
网络输入为784维向量
'''
network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

'''
编译网络
'''
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=10, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:',test_acc)
print('test_loss:',test_loss)

print('the first test image',test_images[0])
print('the first true label',test_labels[0])
print('the first predict label',network.predict(test_images)[0])


