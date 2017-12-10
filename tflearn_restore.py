from __future__ import absolute_import, division, print_function
#import tensorflow as tf
import tflearn
#import tflearn.datasets.mnist as mnist


#X,Y, testX, testY = mnist.load_data(one_hot=True)

input_layer = tflearn.input_data(shape=[None, 784], name='input')
dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
dense2 = tflearn.fully_connected(dense1, 256, name='dense2')

softmax = tflearn.fully_connected(dense2, 10, activation='softmax')

regression = tflearn.regression(softmax, optimizer='adam',\
        learning_rate=0.001,\
        loss='categorical_crossentropy')


model = tflearn.DNN(regression)
#Load a model
model.load("mnist/model.tf1")
#Retrieve a layer weights, by layer name:
# Get a variable's value, using model `get_weights` method:
print("------model method------")
dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
print(model.get_weights(dense1_vars[0]))
print(model.get_weights(dense1.W))
#dense2_vars = tflearn.variables.get_layer_variables_by_name('dense2')
#print("------fuction method------")
#with model.session.as_default():
#    print( tflearn.variables.get_value(dense2_vars[1]))
#    print(  tflearn.variables.get_value(dense2.b))

#model.fit(X, Y, n_epoch=1,
#         validation_set=(testX, testY),
#         show_metric=True,
#         snapshot_epoch=True,
#         batch_size=100,
#         snapshot_step=550,
#         run_id='model_and_weights')

#model.save("mnist/model.tf1")

