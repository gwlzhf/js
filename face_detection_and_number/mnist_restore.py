import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob  
import os


#train_data = pd.read_csv("train.csv")
#train_labels = train_data.pop(item="label")[0:1000]
#train_values = train_data.values[0:1000]
path = 'practice/tfrecord/test/'
list_path = []

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    list_path.append(file_path)

data_set = list_path
# 缩放图像的大小  
IMAGE_HEIGHT = 28  
IMAGE_WIDTH = 28  
# 读取缩放图像  
jpg_data = tf.placeholder(dtype=tf.string)  
decode_jpg = tf.image.decode_jpeg(jpg_data, channels=1)  
resize = tf.image.resize_images(decode_jpg, [IMAGE_HEIGHT, IMAGE_WIDTH])  
resize = tf.cast(resize, tf.uint8)  
def resize_image(file_name):  
    with tf.gfile.FastGFile(file_name, 'rb') as f:  
        image_data = f.read()  
    with tf.Session() as sess:  
        image = sess.run(resize, feed_dict={jpg_data: image_data})  
    return image  
   
#for i in range(10):
#    image_resized = resize_image(data_set[i])
#     print(image_resized)
#    plt.imshow(image_resized)
#    plt.show()



sess=tf.Session()
meta_graph_def = tf.saved_model.loader.load(sess,['mnist_graph'],"practice/tfrecord/model")
x = sess.graph.get_tensor_by_name('x:0')
#y = sess.graph.get_tensor_by_name('y:0')
output = sess.graph.get_tensor_by_name('output/output:0')
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')


#learning_rate = sess.graph.get_tensor_by_name('learning_rate:0')
for i in range(10):
    print("The picture's name is: %s"%data_set[i])
    image = resize_image(data_set[i])
    image_resized = np.reshape(image,[1,784])
    plt.imshow(np.reshape(image_resized,[28,28]))
    plt.show()
    y_pre = np.argmax(sess.run(output,feed_dict={x:image_resized/255,keep_prob:1.0}))
#    if y_pre[i] != train_labels[i]:
    print("when i is %d :The predict is %s ."%(i,y_pre))
sess.close()  
