
# coding: utf-8

# In[1]:


import glob  
import tensorflow as tf
import numpy as np  
from random import shuffle
import os
import matplotlib.pyplot as plt


# In[3]:


age_table=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']  
sex_table=['f','m']  # f:女; m:男  
   
# AGE==True 训练年龄模型，False,训练性别模型  
AGE = True  
   
# if AGE == True:  
age_lables_size = len(age_table) # 年龄  
# else:  
sex_lables_size = len(sex_table) # 性别  
   
# print(lables_size)
face_set_fold = 'dataset/images'  
   
fold_0_data = os.path.join(face_set_fold, 'fold_0_data.txt')  
fold_1_data = os.path.join(face_set_fold, 'fold_1_data.txt')  
fold_2_data = os.path.join(face_set_fold, 'fold_2_data.txt')  
fold_3_data = os.path.join(face_set_fold, 'fold_3_data.txt')  
fold_4_data = os.path.join(face_set_fold, 'fold_4_data.txt')  
   
face_image_set = os.path.join(face_set_fold, 'aligned')


# In[5]:


def parse_data(fold_x_data):  
    data_set = []  
   
    with open(fold_x_data, 'r') as f:  
        line_one = True  
        for line in f:  
            tmp = []  
            if line_one == True:  
                line_one = False  
                continue  
   
            tmp.append(line.split('\t')[0])  
            tmp.append(line.split('\t')[1])  
            tmp.append(line.split('\t')[3])  
            tmp.append(line.split('\t')[4])  
   
            file_path = os.path.join(face_image_set, tmp[0])  
            if os.path.exists(file_path):  
                filenames = glob.glob(file_path + "/*.jpg")  
                for filename in filenames:  
                    if tmp[1] in filename:  
                        break  
#                 if AGE == True:  
                if tmp[2] in age_table:  
                    data_set.append([filename, age_table.index(tmp[2])])  
#                 if tmp[3] in sex_table:  
#                     data_set.append(sex_table.index(tmp[3]))  
   
    return data_set 


# In[34]:


# data_set_0 = parse_data(fold_0_data)
# data_set = parse_data(fold_1_data)  
# data_set = parse_data(fold_2_data)  
# data_set = parse_data(fold_3_data)  
data_set = parse_data(fold_4_data)
# data_set = data_set_0 + data_set_1 + data_set_2 + data_set_3 + data_set_4


# In[30]:



# 缩放图像的大小  
IMAGE_HEIGHT = 227  
IMAGE_WIDTH = 227  
# 读取缩放图像  
jpg_data = tf.placeholder(dtype=tf.string)  
decode_jpg = tf.image.decode_jpeg(jpg_data, channels=3)  
resize = tf.image.resize_images(decode_jpg, [IMAGE_HEIGHT, IMAGE_WIDTH])  
resize = tf.cast(resize, tf.uint8)  
def resize_image(file_name):  
    with tf.gfile.FastGFile(file_name, 'rb') as f:  
        image_data = f.read()  
    with tf.Session() as sess:  
        image = sess.run(resize, feed_dict={jpg_data: image_data})  
    return image  
   


# In[35]:


# sess = tf.InteractiveSession()
# image_resized,age = resize_image(data_set_0[0][0]),data_set_0[0][1]
# print(image_resized)
# print(age)
# plt.imshow(image_resized)
# plt.show()
# sess.close()
writer = tf.python_io.TFRecordWriter(path="data_set_4.tfrecords")
train_size = len(data_set)
for i in range(train_size):
    image_resized,age = resize_image(data_set[i][0]),data_set[i][1]
    image_raw = image_resized.tostring()

    example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[age]))
                    }))
    writer.write(record=example.SerializeToString())
writer.close()

