import tensorflow as tf  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import csv  
import os  
  
os.chdir("/Users/apple/Desktop/")  
print(os.getcwd())  
  
def read_data(file_queue):  
    reader = tf.TextLineReader(skip_header_lines=0)  
    key, value = reader.read(queue=file_queue)  
    defaults = list([0.] for i in range(17))  
    input_Idq_s_1, input_Idq_s_2, input_Edq_s_1, input_Edq_s_2, input_MagV_s, input_Idq_m_1, input_Idq_m_2, \  
        input_Edq_m_1, input_Edq_m_2, input_MagV_m, input_Vdc, input_Pt, output_Vdq_s_1, output_Vdq_s_2, \  
        output_Vdq_m_1, output_Vdq_m_2, rand = tf.decode_csv(records=value, record_defaults=defaults)  
    input_list = list([input_Idq_s_1, input_Idq_s_2, input_Edq_s_1, input_Edq_s_2, input_MagV_s, input_Idq_m_1, \  
        input_Idq_m_2, input_Edq_m_1, input_Edq_m_2, input_MagV_m, input_Vdc, input_Pt])  
    output_list = list([output_Vdq_s_1, output_Vdq_s_2, output_Vdq_m_1, output_Vdq_m_2])  
    input_minimum = list([26.9223227068843, -98.4780345017635, 94.2182270883746, -2547.04028098514, 188.434752223462,   
        -3074.24987628734, -80.0663030792083, 94.3688439437233, -4294.32895768398, 188.737903943159, 11.3363750371760, \  
        14.5698874167718])  
    input_maximum = list([338.471317085834, 132.425043875557, 8178.66679759072, 3962.04092754847, 14196.4064943722, \  
        1551.68716264095, 75.9558804677249, 8170.27398930453, 4158.28883844735, 14515.0865307378, 20098.8974807069, \  
        12123066.0740678])  
    output_minimum = list([-105.675264765919, -3839.50483890428, -9675.45018951087, -3704.86493155417])  
    output_maximum = list([10981.3824330706, 5832.43660916112, 105.536592510222, 8641.08573453797])  
    input_normalization = list(map(lambda x, min_x, max_x: (x - min_x) / (max_x - min_x), input_list, \  
        input_minimum, input_maximum))  
    output_normalization = list(map(lambda x, min_x, max_x: (x - min_x) / (max_x - min_x), output_list, \  
        output_minimum, output_maximum))  
    return input_normalization, output_normalization[3]  
  
def create_pipeline(filename, batch_size, num_epochs=None):  
    file_queue = tf.train.string_input_producer(string_tensor=[filename], num_epochs=num_epochs)  
    example, label = read_data(file_queue=file_queue)  
    min_after_dequeue = 1000  
    capacity = min_after_dequeue + batch_size  
    example_batch, label_batch = tf.train.shuffle_batch(  
        tensors=[example, label],  
        batch_size=batch_size,  
        capacity=capacity,  
        min_after_dequeue=min_after_dequeue  
    )  
    return example_batch, label_batch  
  
def relu(x, alpha=0., max_value=None):  
    negative_part = tf.nn.relu(-x)  
    x = tf.nn.relu(x)  
    if max_value is not None:  
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),  
                             tf.cast(max_value, dtype=tf.float32))  
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part  
    return x  
  
def add_layer(inputs, in_size, out_size, activation_function=None, layer_name='Layer'):  
    with tf.name_scope(layer_name):  
        with tf.name_scope('Weights'):  
            Weights = tf.Variable(initial_value=tf.random_normal([in_size, out_size]))  
        with tf.name_scope('Biases'):  
            biases = tf.Variable(initial_value=tf.zeros(shape=[1, out_size]) + 0.1)  
        with tf.name_scope('Wx_plus_b'):  
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  
        if activation_function is None:  
            outputs = Wx_plus_b  
        elif activation_function == tf.nn.relu:  
            outputs = relu(Wx_plus_b, alpha=0.05)  
        else:  
            outputs = activation_function(Wx_plus_b)  
        return outputs, Weights, biases  
  
x_train_batch, y_train_batch = create_pipeline(filename='m3c_data_train.csv', batch_size=50, num_epochs=1000)  
x_test, y_test = create_pipeline(filename='m3c_data_test.csv', batch_size=60)  
global_step = tf.Variable(initial_value=0, trainable=False)  
#learning_rate = tf.constant(1e-3, dtype=tf.float32)  
learning_rate = tf.train.exponential_decay(1e-4, global_step, 1e4, 1e-5)  
  
with tf.name_scope('InputLayer'):  
    x = tf.placeholder(dtype=tf.float32, shape=[None, 12])  
    y = tf.placeholder(dtype=tf.float32, shape=[None])  
      
hidden_0, Weights_hidden_0, biases_hidden_0 = add_layer(inputs=x, in_size=12, out_size=5, \  
    activation_function=tf.nn.relu, layer_name='HiddenLayer0')  
hidden_1, Weights_hidden_1, biases_hidden_1 = add_layer(inputs=hidden_0, in_size=5, out_size=5, \  
    activation_function=tf.nn.relu, layer_name='HiddenLayer1')  
prediction, Weights_prediction, biases_prediction = add_layer(inputs=hidden_1, in_size=5, out_size=1, \  
    activation_function=None, layer_name='OutputLayer')  
  
with tf.name_scope('Loss'):  
    loss = tf.reduce_mean(tf.reduce_sum(input_tensor=tf.square(y-prediction), reduction_indices=[1]))  
tf.summary.scalar(name='Loss', tensor=loss)  
  
with tf.name_scope('Train'):  
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, \  
        global_step=global_step)  
  
correct_prediction = tf.equal(tf.argmax(input=prediction, axis=1), tf.cast(y, tf.int64))  
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))  
tf.summary.scalar(name='Accuracy', tensor=accuracy)  
  
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  
merged_summary = tf.summary.merge_all()  
  
sess = tf.Session()  
train_writer = tf.summary.FileWriter(logdir='logs/train', graph=sess.graph)  
test_writer = tf.summary.FileWriter(logdir='logs/test', graph=sess.graph)  
sess.run(init)  
  
coord = tf.train.Coordinator()  
threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
  
try:  
    print("Training: ")  
    count = 0  
    curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])  
  
    while not coord.should_stop():  
        count += 1  
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])  
        sess.run(train_step, feed_dict={  
            x: curr_x_train_batch, y: curr_y_train_batch  
        })  
        lr = sess.run(learning_rate)  
        loss_, summary = sess.run([loss, merged_summary], feed_dict={  
            x: curr_x_train_batch, y: curr_y_train_batch  
        })  
        train_writer.add_summary(summary, count)  
        loss_, test_acc, test_summary = sess.run([loss, accuracy, merged_summary], feed_dict={  
            x: curr_x_test_batch, y: curr_y_test_batch  
        })  
        test_writer.add_summary(summary=summary, global_step=count)  
        print('Batch =', count, 'LearningRate =', lr, 'Loss =', loss_)  
  
except tf.errors.OutOfRangeError:  
    print('Done training -- epoch limit reached')  
  
finally:  
    writer = tf.summary.FileWriter("logs/", sess.graph)  
    Weights_hidden_0_, biases_hidden_0_, Weights_hidden_1_, biases_hidden_1_, Weights_prediction_, \  
        biases_prediction_ = sess.run([Weights_hidden_0, biases_hidden_0, Weights_hidden_1, biases_hidden_1, \  
        Weights_prediction, biases_prediction])  
    export_data = pd.DataFrame(data=list(Weights_hidden_0_)+list(biases_hidden_0_)+list(Weights_hidden_1_)+\  
        list(biases_hidden_1_)+list(Weights_prediction_)+list(biases_prediction_))  
    export_data.to_csv('results_output_3.csv')  
  
coord.request_stop()  
coord.join(threads=threads)  
