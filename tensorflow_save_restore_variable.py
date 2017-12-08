import tensorflow as tf
'''
#Creat some varuable
v1 = tf.get_variable("v1",shape=[3],initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2",shape=[5],initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

#Add an op to initialize the variable
init_op = tf.global_variables_initializer()


#Add ops to save and restore all the variables.
saver = tf.train.Saver()

#Later, launch the model, initialize the variable,do some work and save the 
#the variable to disk.
with tf.Session() as sess:
    sess.run(init_op)
    print(v1.eval())
    print(v2.eval())
    inc_v1.op.run()
    dec_v2.op.run()
    #Save the variable to disk
    save_path = saver.save(sess,"/tmp/save.ckpt")
    print("model saved in file %s" % save_path )

'''
#Restore
tf.reset_default_graph()
#creat the variale that you want to restore
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

#Add ops to retore variable
saver = tf.train.Saver([v2])

#use the saver object normally after that.
with tf.Session() as sess:
    #v1.initializer.run()
    #v2.initializer.run()
    #initialize v1 since the saver will not
    saver.restore(sess,"/tmp/save.ckpt")
    #print("v1 : %s" %v1.eval())
    print("v2 : %s" %v2.eval())
