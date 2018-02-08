#总的图片数17393张
# coding: utf-8
import glob  
import tensorflow as tf
import numpy as np  
from random import shuffle
import os
import matplotlib.pyplot as plt
slim = tf.contrib.slim


filename_queue = tf.train.string_input_producer(["data_set.tfrecords"],num_epochs=5)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            })
images = tf.reshape(tf.decode_raw(features['image_raw'],tf.uint8),[227,227,3])/255

labels = tf.one_hot(tf.cast(features['label'], tf.int32),2)
image_size = images.shape[0]
min_after_dequeue = 1000
batch_size = 32
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.shuffle_batch([images,labels],batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)


# In[4]:


X = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3],name='x')  
Y = tf.placeholder(dtype=tf.float32, shape=[None,2],name='y')  
keep_prob = tf.placeholder(dtype=tf.float32,shape=[],name='keep_prob')
   
def conv_net(nlabels, images):  
#    weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)  
    with tf.name_scope("conv_net") as scope:  
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = tf.nn.relu,biases_initializer=tf.constant_initializer(0.), weights_initializer=tf.random_normal_initializer(stddev=0.1), trainable=True):
            conv1 = slim.conv2d(images, 32, [7,7], [4, 4], scope='conv1')  
            pool1 = slim.max_pool2d(conv1, [3,3], stride=2, scope='pool1') 
            norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')   
            conv2 = slim.conv2d(norm1, 128, [5, 5], [1, 1], padding='SAME', scope='conv2')   
            pool2 = slim.max_pool2d(conv2, [3,3], stride=2, padding='VALID', scope='pool2')
            norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')    
            conv3 = slim.conv2d(norm2, 32, [3, 3], [1, 1], padding='SAME', scope='conv3')  
            pool3 = slim.max_pool2d(conv3, [3,3], stride=2, padding='VALID', scope='pool3')  
#             flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')  
            flat = slim.flatten(pool3,scope='flatten')
            full1 = slim.fully_connected(flat,256, scope='full1')  
            drop1 = slim.dropout(full1, keep_prob,scope='drop1')  
            full2 = slim.fully_connected(drop1, 1024, scope='full2')  
            drop2 = slim.dropout(full2, keep_prob, scope='drop2')  
    with tf.name_scope('output') as scope:  
        weights = tf.Variable(tf.random_normal([1024, nlabels], mean=0.0, stddev=0.1), name='weights')  
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')  
        output = tf.add(tf.matmul(drop2, weights), biases, name="output")
    return output
  

y_ = conv_net(2, X)  
total_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,y_))

optimizer = tf.train.AdamOptimizer(1e-3)
train_op = slim.learning.create_train_op(total_loss, optimizer)

correct_prediction = tf.equal( tf.argmax(Y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean( tf.cast(correct_prediction,tf.float16))


#saver = tf.train.Saver(tf.global_variables())
init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
sess=tf.Session()
sess.run(init_op)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)


for i in range(500):  
    batch_x, batch_y = sess.run([image_batch, label_batch])
    loss,loss = sess.run([train_op,total_loss], feed_dict={X:batch_x, Y:batch_y,keep_prob:0.5})
    if i % 10 == 0 :
        print("loss is %f .step %d ,accuracy is:%f"%(loss,i/10,sess.run(accuracy,feed_dict={X:batch_x, Y:batch_y,keep_prob:1.0})))
print("training has been finished!") 
#build a saver for model
builder = tf.saved_model.builder.SavedModelBuilder("model/")
builder.add_meta_graph_and_variables(sess, ['csv_graph'])
builder.save()
print("model has been saved!")
 
#saver.save(sess, 'module/model')  
coord.request_stop()  
coord.join(threads)
sess.close()   


# In[47]:

'''
init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
sess=tf.InteractiveSession()
sess.run(init_op)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(10):
    [image,label] = sess.run([images,labels])
    plt.imshow(image)
    plt.show()
    print(label)
coord.request_stop()  
coord.join(threads)
sess.close()
'''
