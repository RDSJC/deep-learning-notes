from keras.models import *
from keras.preprocessing import *
from keras import *
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K 
from keras.applications import VGG16
from keras.applications.vgg16 import *

model = load_model('cats_vs_dogs_small_origin.h5')
model.summary()

####################
'''可视化中间激活 '''
####################

img_path = 'dataset/small_data/test/cats/cat.1513.jpg'
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img) #形状 (150,150,3)
img_tensor = np.expand_dims(img_tensor, axis=0) #表示在0位置添加数据 (150,150,3) -> (1,150,150,3)
img_tensor /= 255.
print(img_tensor.shape) #(1,150,150,3)

plt.imshow(img_tensor[0])
plt.show()

'''输入一张图像，这个模型将返回原始模型前8层的激活值。 '''
layer_outputs = [layer.output for layer in model.layers[:8]] #提取前8层的输出
activation_model = models.Model(inputs=model.input,outputs=layer_outputs) #创建一个模型，给定模型输入，可以返回这些输出

'''返回8个numpy数组组成的列表，每个层激活对应一个numpy数组 '''
activations = activation_model.predict(img_tensor) #每层的激活
first_layer_activation = activations[0] #第一个卷积层的激活
print(first_layer_activation.shape) #(1,148,148,32) 148x148的特征图，有32个通道
plt.imshow(first_layer_activation[0, :, :, 4])#将第四个通道可视化
plt.show()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names,activations): #显示特征图
    n_features = layer_activation.shape[-1] #特征图中特征的个数
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row #将激活通道平铺
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
            row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
            scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

###############################
'''可视化卷积神经网络的过滤器 '''
###############################

model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block3_conv1'
filter_index = 0

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern(layer_name , filter_index))
plt.show()

#########################
'''可视化类激活的热力图 '''
#########################
model = load_model('cats_vs_dogs_small_origin.h5')

img_path = 'dataset/small_data/test/cats/cat.1513.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
print(preds)
#TODO

