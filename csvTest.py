import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#read csv file,including label 1 and pixel 784.
train_data = pd.read_csv("train.csv")
x_pixel = np.array(train_data.iloc[:,1:],dtype=np.float16)
y_label = np.array(train_data.iloc[:,0])

test_data = pd.read_csv("test.csv")
test_x_pixel = np.array(test_data.iloc[:,0:],dtype=np.float16)
#transform y_label from (1,N) to (N,class) ,one hot vector.
enc = OneHotEncoder()
y_onehot = enc.fit_transform(y_label.reshape(-1,1)).toarray()

#split the training data to two part.one for training,the other for testing
train_x,test_x,train_y,test_y = train_test_split(\
        x_pixel,y_onehot,test_size=0.1,random_state=0)


#define the placeholder for the input and output
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

#define the weights and bias
#W = tf.truncated_normal(shape=[784,10],stddev=1.0,name='weights')
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#define matrix convolve
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#first layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#first fully connecting
W_full1 = weight_variable([7,7,64,1024])
b_full1 = bias_variable([1024])

h_full1 = tf.reshape(tf.nn.relu( tf.nn.conv2d(h_pool2,W_full1,strides=[1,1,1,1],padding='VALID') + b_full1),[-1,1024])
#dropout layer
keep_prob = tf.placeholder(tf.float32)
full_dropout = tf.nn.dropout(h_full1,keep_prob)

#second fully connecting
W_full2 = weight_variable([1024,10])
b_full2 = bias_variable([10])

#W = tf.Variable( tf.truncated_normal(([784, 10]),stddev=0.1))
#b = tf.Variable(tf.zeros([10]))
y = tf.matmul(full_dropout,W_full2) + b_full2

#compute the loss
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal( tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean( tf.cast(correct_prediction,tf.float16)) 
 
#create a session and grah
#initial all the variable
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

train_size = train_x.shape[0]
for i in range(500):
    start = i*32 % train_size
    end = (i+1) % train_size
    if start > end:
        start = 0
    batch_x = train_x[start:end]
    batch_y = train_y[start:end]
    sess.run(train_step,feed_dict={x:batch_x,y_:batch_y,keep_prob:0.6})
    if i % 10 == 0 :
        print("%d accuracy is: %f"%(i,sess.run(accuracy,feed_dict={x:test_x[0:50], y_: test_y[0:50] ,keep_prob:1.0})))

f=open("submission.csv", "a+")
f.write("ImageId,Label" + "\n")
for j in range(28):
    start = 1000*j
    test = y.eval(feed_dict={x:test_x_pixel[start:start+1000],keep_prob:1.0})
    test_temp = np.argmax(test,axis=1)
    for i in range(test_temp.shape[0]):
        new_context = str(1000*j+i+1) + "," + str(test_temp[i]) + '\n'
        f.write(new_context)
f.close()
sess.close()


#print(train_x.shape)
#print(y_label.reshape(-1,1).shape)
#print(y_onehot[0:10])


