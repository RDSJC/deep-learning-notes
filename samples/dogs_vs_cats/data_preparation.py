import os
import shutil

'''原始数据目录'''
original_dataset_dir = 'dataset/original_data'
original_dataset_train_dir = os.path.join(original_dataset_dir,'train')
original_dataset_test_dir = os.path.join(original_dataset_dir,'test')
'''保存较小数据集的目录'''
base_dir = 'dataset/small_data'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

'''训练集目录'''
train_dir = os.path.join(base_dir,'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
'''验证集目录'''
val_dir = os.path.join(base_dir,'val')
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
'''测试集目录'''
test_dir = os.path.join(base_dir,'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

'''猫的训练图像目录'''
train_cats_dir = os.path.join(train_dir,'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)
'''狗的训练图像目录'''
train_dogs_dir = os.path.join(train_dir,'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

'''猫的验证图像目录'''
val_cats_dir = os.path.join(val_dir,'cats')
if not os.path.exists(val_cats_dir):
    os.mkdir(val_cats_dir)
'''狗的训练图像目录'''
val_dogs_dir = os.path.join(val_dir,'dogs')
if not os.path.exists(val_dogs_dir):
    os.mkdir(val_dogs_dir)

'''猫的测试图像目录'''
test_cats_dir = os.path.join(test_dir,'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)
'''狗的测试图像目录'''
test_dogs_dir = os.path.join(test_dir,'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)


'''将前1000张猫的图像复制到train_cats_dir'''
fnames = []
for i in range(1000):
    fnames.append('cat.{}.jpg'.format(i))
for fname in fnames:
    src = os.path.join(original_dataset_train_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
'''将接下来500张猫的图像复制到val_cats_dir'''
fnames = []
for i in range(1000,1500):
    fnames.append('cat.{}.jpg'.format(i))
for fname in fnames:
    src = os.path.join(original_dataset_train_dir,fname)
    dst = os.path.join(val_cats_dir,fname)
    shutil.copyfile(src,dst)
'''将接下来500张猫的图像复制到test_cats_dir'''
fnames = []
for i in range(1500,2000):
    fnames.append('cat.{}.jpg'.format(i))
for fname in fnames:
    src = os.path.join(original_dataset_train_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)


'''将前1000张狗的图像复制到train_dogs_dir'''
fnames = []
for i in range(1000):
    fnames.append('dog.{}.jpg'.format(i))
for fname in fnames:
    src = os.path.join(original_dataset_train_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)
'''将接下来500张狗的图像复制到val_dogss_dir'''
fnames = []
for i in range(1000,1500):
    fnames.append('dog.{}.jpg'.format(i))
for fname in fnames:
    src = os.path.join(original_dataset_train_dir,fname)
    dst = os.path.join(val_dogs_dir,fname)
    shutil.copyfile(src,dst)
'''将接下来500张狗的图像复制到test_dog_dir'''
fnames = []
for i in range(1500,2000):
    fnames.append('dog.{}.jpg'.format(i))
for fname in fnames:
    src = os.path.join(original_dataset_train_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)

