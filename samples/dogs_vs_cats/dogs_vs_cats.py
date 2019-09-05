from  keras import *
from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import numpy as np
from keras.models import *

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) #降低过拟合，让训练集犯错，增强泛化能力
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    return model

'''将所有图像像素乘以1/255'''
train_dir = 'dataset/small_data/train'
val_dir = 'dataset/small_data/val'
'''数据增强'''
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40, #随机旋转40度以内
    width_shift_range = 0.2, #图像在水平方向上平移的范围(相对于总宽度的比例)
    height_shift_range = 0.2, #图像在垂直方向上平移的范围(相对于总高度的比例)
    shear_range = 0.2, #随机错切变换的角度
    zoom_range = 0.2, #图像随机缩放的范围
    horizontal_flip = True #随机将一半图像水平翻转，如果没有水平不对称的假设（比如真实世界的图像），这种做法是有意义的
)
#验证集不能增强
val_datagen = ImageDataGenerator(rescale=1./255)

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

'''
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
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
model.save('cats_vs_dogs_small_origin.h5')

'''绘制图像'''
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

'''test'''
model = load_model('C:/Users/ShiJC/Desktop/learning/dogs_vs_cats/cats_vs_dogs_small.h5')

img_path = 'C:/Users/ShiJC/Desktop/learning/dogs_vs_cats/dataset/small_data/test/cats/cat.1513.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
print(preds)