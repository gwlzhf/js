import tensorflow as tf
import numpy as np


age_table=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']  
filename_queue = tf.train.string_input_producer(["data_set.tfrecords"])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            })
images = tf.reshape(tf.decode_raw(features['image_raw'],tf.uint8),[227,227,3]) 

labels = tf.one_hot(tf.cast(features['label'], tf.int32),8)

min_after_dequeue = 1000
batch_size = 10 
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.batch([images,labels],batch_size=batch_size,capacity=capacity)




init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
sess=tf.Session()
sess.run(init_op)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)
meta_graph_def = tf.saved_model.loader.load(sess,['csv_graph'],"model")
x = sess.graph.get_tensor_by_name('x:0')
#y = sess.graph.get_tensor_by_name('y:0')
output = sess.graph.get_tensor_by_name('output/output:0')
prob = sess.graph.get_tensor_by_name('keep_prob:0')
for i in range(10):
#   print("Person's age is" + age_table[y_list[i]])
    test_image = sess.run(image_batch)
    y_list = np.argmax(sess.run(output,feed_dict={x:test_image,prob:1.0}),axis=1)
    print(y_list)
coord.request_stop()  
coord.join(threads)
sess.close()  
