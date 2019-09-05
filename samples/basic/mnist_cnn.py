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
数据归一化，0-1
标签编码，如：数字3的编码为[0,0,0,1,0,0,0,0,0,0]
'''
train_images = train_images.reshape((train_images.shape[0],28,28,1))
test_images = test_images.reshape((test_images.shape[0],28,28,1))

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
'''
model = models.Sequential()

#每个con2d和maxpooling2d层的输出都是一个形状为(height,width,channels)的3D张量
#宽度和高度尺寸会随着网络加深而变小，通道数量由传入conv2d层的第一个参数所控制
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1))) #32个3*3的卷积核
model.add(layers.MaxPool2D((2,2))) 
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten()) #将3D张量输出展平为1D
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


'''
编译网络
'''
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:',test_acc)
print('test_loss:',test_loss)

print('the first test image',test_images[0])
print('the first true label',test_labels[0])
print('the first predict label',model.predict(test_images)[0])


