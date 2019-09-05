from  keras import *
from keras.preprocessing.image import *
import matplotlib.pyplot as plt
from keras.applications import VGG16
import numpy as np

'''
如果想要将深度学习应用于小型图像数据集，一种常用且非常高效的方法是使用预训练网络
使用预训练网络有两种方法：特征提取(feature extraction)和微调模型(fine-tuning)
'''

'''
特征提取是使用之前网络学到的表示来从新样本中提取出有趣的特征，然后将这些特征输入一个新的分类器，从头开始训练
卷积神经网络包含两部分：一系列的卷积层和池化层（卷积基 convolutional base），最后是一个密集连接分类器
对于卷积神经网络而言，特征提取就是取出之前训练好的网络的卷积基，在上面运行新数据，然后在输出上面训练一个新的分类器

为什么仅重复使用卷积基?
1.原因在于卷积基学到的表示可能更加通用，因此更适合重复使用。但是，分类器学到的表示必然是针对于模型训练的类别，其中仅包含某个类别出现在整
张图像中的概率信息。
2.此外，密集连接层的表示不再包含物体在输入图像中的位置信息。密集连接层舍弃了空间的概念，而物体位置信息仍然由卷积特征图所描述。如果物体位置对于问题
很重要，那么密集连接层的特征在很大程度上是无用的。
'''

train_dir = 'dataset/small_data/train'
val_dir = 'dataset/small_data/val'
test_dir = 'dataset/small_data/test'

def curve(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


'''
不使用数据增强的快速特征提取
首先，运行ImageDataGenerator 实例，将图像及其标签提取为Numpy 数组。我们需要
调用conv_base 模型的predict 方法来从这些图像中提取特征。
'''
'''
def extract_features(directory, sample_count):
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
    features = np.zeros((sample_count,4,4,512)) #conv_base最后一层输出为(4,4,512)
    labels = np.zeros((sample_count,)) 
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20
    generator = datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size = batch_size,
        class_mode = 'binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir,2000)
val_features, val_labels = extract_features(val_dir,1000)
test_features, test_labels = extract_features(test_dir,1000)

train_features = train_features.reshape(2000,4*4*512)
val_features = val_features.reshape(1000,4*4*512)
test_features = test_features.reshape(1000,4*4*512)

model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_shape=(4*4*512,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])
history = model.fit(train_features, train_labels,epochs=30,batch_size=20,validation_data=(val_features, val_labels))
curve(history)
'''


'''
使用数据增强的特征提取
扩展conv_base 模型，然后在输入数据上端到端地运行模型。
'''
def create_model():
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3)) #weights指定模型初始化的权重检查点 include_top指定模型最后是都包含密集连接分类器
    conv_base.trainable = False #冻结卷积基在训练过程中保持其权重不变，如果不这么做，那么卷积基之前学到的表示将会在训练过程中被修改，因为其上添加的Dense层是随机初始化的，所以非常大的权重更新将会在网络中传播，对之前学到的表示造成很大破坏
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) #降低过拟合，让训练集犯错，增强泛化能力
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
    return model


'''数据处理与增强'''
train_datagen = ImageDataGenerator(
    rescale = 1./255,#将所有图像像素乘以1/255
    rotation_range = 40, #随机旋转40度以内
    width_shift_range = 0.2, #图像在水平方向上平移的范围(相对于总宽度的比例)
    height_shift_range = 0.2, #图像在垂直方向上平移的范围(相对于总高度的比例)
    shear_range = 0.2, #随机错切变换的角度
    zoom_range = 0.2, #图像随机缩放的范围
    horizontal_flip = True #随机将一半图像水平翻转，如果没有水平不对称的假设（比如真实世界的图像），这种做法是有意义的
)
#验证集不能增强
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

'''数据生成'''
train_generator = train_datagen.flow_from_directory(
    train_dir, #目标目录的路径。每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。
    target_size = (150,150), #将所有图像的大小调整为150x150
    batch_size = 20,
    class_mode = 'binary' #因为使用了binary_crossentropy损失，所以需要二进制标签
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

'''
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)    #(batch_size,width,height,channel)
    print('label batch shape:', labels_batch.shape)
    print(labels_batch)
    break
'''

'''训练模型'''
model = create_model()
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 50
)
model.save('cats_vs_dogs_small.h5')
'''绘制图像'''
curve(history)
'''在测试集上评估'''
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)  #0.88499
print('test loss:', test_loss)  #0.2658

