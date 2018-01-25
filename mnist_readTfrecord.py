import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
slim = tf.contrib.slim

test_data = pd.read_csv("test.csv")
test_x_pixel = np.array(test_data.iloc[:,0:],dtype=np.float16)

filename_queue = tf.train.string_input_producer(["train.tfrecords"],num_epochs=60)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            })
images = tf.reshape(tf.decode_raw(features['image_raw'],tf.int64),[784])/255

labels = tf.one_hot(tf.cast(features['label'], tf.int32),10)
#image_size = images.shape[0]

min_after_dequeue = 1000
batch_size = 50 
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.shuffle_batch([images,labels],batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)

#create the graph
x = tf.placeholder(tf.float32,shape=[None,784],name="x")
y_ = tf.placeholder(tf.float32,shape=[None,10],name="y_")
keep_prob = tf.placeholder(dtype=tf.float32,shape=[],name='keep_prob')
learning_rate = tf.placeholder(dtype=tf.float32,shape=[],name='learning_rate')
x_images = tf.reshape(x,[-1,28,28,1])

def conv_net(nlabels, x_images):  
    with tf.name_scope("conv_net") as scope:
        with slim.arg_scope([slim.conv2d,slim.fully_connected], activation_fn = tf.nn.relu,weights_initializer = tf.truncated_normal_initializer(stddev=0.01),biases_initializer=tf.constant_initializer(0.),trainable=True):
                net = slim.conv2d(x_images, 32, [5,5],stride=1,padding='SAME',scope='conv1')
                net = slim.max_pool2d(net, [2,2],stride=2,padding='SAME',scope='pool1')
                net = tf.nn.local_response_normalization(net,name="response")
                net = slim.conv2d(net, 64, [3,3],stride=1,padding='SAME',scope='conv3')
                net = slim.max_pool2d(net, [2,2], stride=2,padding='SAME',scope='pool2')
                net = slim.flatten(net,scope='flatten')
                net = slim.fully_connected(net, 1024, scope='full1')
                net = slim.dropout(net, keep_prob, scope = 'dropout1')
    #        y = slim.fully_connected(net, 10,scope='output')
    with tf.name_scope('output') as scope:  
         weights = tf.Variable(tf.random_normal([1024, 10], mean=0.0, stddev=0.01), name='weights')  
         biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), name='biases')  
         output = tf.add(tf.matmul(net, weights), biases, name="output")   
    return output

#compute the loss
y = conv_net(10,x_images)
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = slim.learning.create_train_op(cross_entropy, optimizer)
correct_prediction = tf.equal( tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean( tf.cast(correct_prediction,tf.float16)) 
 
#create a session and grah
#initial all the variable
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(15000):
    image_batchs,label_batchs = sess.run([image_batch,label_batch])
    if i < 4000:
        sess.run(train_step,feed_dict={x:image_batchs,y_:label_batchs,learning_rate:1e-3,keep_prob:0.5})
    elif i < 10000:
        sess.run(train_step,feed_dict={x:image_batchs,y_:label_batchs,learning_rate:1e-4,keep_prob:0.5})
    else:
        sess.run(train_step,feed_dict={x:image_batchs,y_:label_batchs,learning_rate:1e-5,keep_prob:0.5})

    if i % 10 == 0 :
        print("%d accuracy is: %f"%(i,sess.run(accuracy,feed_dict={x:image_batchs, y_: label_batchs,keep_prob:1.0})))

#build a saver for model
builder = tf.saved_model.builder.SavedModelBuilder("model/")
builder.add_meta_graph_and_variables(sess, ['mnist_graph'])
builder.save()
print("model has been saved!")

#save_path = saver.save(sess, "/tmp/save.cpkt")
#print("model save in file %s" %save_path)
f=open("submission.csv", "a+")
f.write("ImageId,Label" + "\n")
for j in range(28):
    start = 1000*j
    test = y.eval(session=sess,feed_dict={x:test_x_pixel[start:start+1000]/255,keep_prob:1.0})
    test_temp = np.argmax(test,axis=1)
    for i in range(test_temp.shape[0]):
        new_context = str(1000*j+i+1) + "," + str(test_temp[i]) + '\n'
        f.write(new_context)
f.close()
coord.request_stop()  
coord.join(threads) 
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
