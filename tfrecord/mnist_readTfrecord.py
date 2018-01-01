import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test_data = pd.read_csv("test.csv")
test_x_pixel = np.array(test_data.iloc[:,0:],dtype=np.float16)
filename_queue = tf.train.string_input_producer(["train.tfrecords"])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            })
images = tf.reshape(tf.decode_raw(features['image_raw'],tf.int64),[784])

labels = tf.one_hot(tf.cast(features['label'], tf.int32),10)
image_size = images.shape[0]

min_after_dequeue = 1000
batch_size = 16 
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.batch([images,labels],batch_size=batch_size, capacity=capacity)

#create the graph
x = tf.placeholder(tf.float32,shape=[None,784],name="x")
y_ = tf.placeholder(tf.float32,shape=[None,10],name="y_")

#define the weights and bias
#W = tf.truncated_normal(shape=[784,10],stddev=1.0,name='weights')
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name="weight")
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name="bias")

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
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
full_dropout = tf.nn.dropout(h_full1,keep_prob)

#second fully connecting
W_full2 = weight_variable([1024,10])
b_full2 = bias_variable([10])

#W = tf.Variable( tf.truncated_normal(([784, 10]),stddev=0.1))
#b = tf.Variable(tf.zeros([10]))
y = tf.add(tf.matmul(full_dropout,W_full2) , b_full2,name="y")

#compute the loss
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal( tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean( tf.cast(correct_prediction,tf.float16)) 
 
#create a session and grah
#initial all the variable
sess = tf.Session()
#build a saver for variables
#saver = tf.train.Saver([W_conv1,b_conv1])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
init = tf.global_variables_initializer()
sess.run(init)

for i in range(5000):
    [image_batchs,label_batchs] = sess.run([image_batch,label_batch])
    sess.run(train_step,feed_dict={x:image_batchs,y_:label_batchs,keep_prob:0.5})
    if i % 10 == 0 :
        print("%d accuracy is: %f"%(i,sess.run(accuracy,feed_dict={x:image_batchs, y_: label_batchs,keep_prob:1.0})))

#build a saver for model
#builder = tf.saved_model.builder.SavedModelBuilder("model/")
#builder.add_meta_graph_and_variables(sess, ['csv_graph'])
#builder.save()
coord.request_stop()  
coord.join(threads) 
print("model has been saved!")

#save_path = saver.save(sess, "/tmp/save.cpkt")
#print("model save in file %s" %save_path)
f=open("submission.csv", "a+")
f.write("ImageId,Label" + "\n")
for j in range(28):
    start = 1000*j
    test = y.eval(session=sess,feed_dict={x:test_x_pixel[start:start+1000],keep_prob:1.0})
    test_temp = np.argmax(test,axis=1)
    for i in range(test_temp.shape[0]):
        new_context = str(1000*j+i+1) + "," + str(test_temp[i]) + '\n'
        f.write(new_context)
f.close()
sess.close()
'''
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image,label = sess.run([images,labels])
    print(label)
    plt.imshow(image)
    plt.show()
coord.request_stop()  
coord.join(threads) 
sess.close()
'''
