import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#hello = tf.constant('hello, Tensorflow!')
#使用占位符placeholder进行变量输入输出
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
#矩阵相乘
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
x = tf.Variable(0,dtype=tf.int16)


#定义一些操作
add = tf.add(a,b)
mul = tf.multiply(a,b)
product = tf.matmul(matrix1,matrix2)
assign_op = tf.assign(x,x+1)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #运行的每一个操作，把变量输进去
    print("变量相加： %i" %sess.run(add,feed_dict={a:2,b:3}))
    print("变量相乘: %i" %sess.run(mul,feed_dict={a:2,b:3}))
    print("矩阵相乘: %i" %sess.run(product)) 

with tf.Session() as sess:
    sess.run(init)
    for x in range(4):
        print(sess.run(assign_op))
        sess.run(assign_op)



