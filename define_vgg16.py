import tensorflow as tf
import tf.contrib.slim as slim


vgg = tf.contrib.slim.nets.vgg  
def vgg16(inputs):  
  with slim.arg_scope([slim.conv2d, slim.fully_connected],  
                      activation_fn=tf.nn.relu,  
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),  
                      weights_regularizer=slim.l2_regularizer(0.0005)):  
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')  
    net = slim.max_pool2d(net, [2, 2], scope='pool1')  
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')  
    net = slim.max_pool2d(net, [2, 2], scope='pool2')  
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')  
    net = slim.max_pool2d(net, [2, 2], scope='pool3')  
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')  
    net = slim.max_pool2d(net, [2, 2], scope='pool4')  
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')  
    net = slim.max_pool2d(net, [2, 2], scope='pool5')  
    net = slim.fully_connected(net, 4096, scope='fc6')  
    net = slim.dropout(net, 0.5, scope='dropout6')  
    net = slim.fully_connected(net, 4096, scope='fc7')  
    net = slim.dropout(net, 0.5, scope='dropout7')  
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')  
  return net  

关于loss,要说一下定义自己的loss的方法，以及注意不要忘记加入到slim中让slim看到你的loss。
还有正则项也是需要手动添加进loss当中的，不然最后计算的时候就不优化正则目标了。
[python] view plain copy
# Load the images and labels.  
images, scene_labels, depth_labels, pose_labels = ...  
  
# Create the model.  
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)  
  
# Define the loss functions and get the total loss.  
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)  
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)  
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)  
slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.  
  
# The following two ways to compute the total loss are equivalent:  
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())  
total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss  
  
# (Regularization Loss is included in the total loss by default).  
total_loss2 = slim.losses.get_total_loss()  
# Load the images and labels.  
images, labels = ...  
  
# Create the model.  
predictions, _ = vgg.vgg_16(images)  
  
# Define the loss functions and get the total loss.  
loss = slim.losses.softmax_cross_entropy(predictions, labels)  

