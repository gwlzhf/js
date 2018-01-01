import tensorflow as tf
import numpy as np
import pandas as pd

#load .csv file
train_data = pd.read_csv("train.csv")
train_labels = train_data.pop(item="label")
train_values = train_data.values

#print(train_labels.shape)
#print(train_values.shape[1])

#---------------create TFRecord file------------------------------
writer = tf.python_io.TFRecordWriter(path="h.tfrecords")
train_size = train_values.shape[0]
for i in range(1):
    image_raw = train_values[i].tostring()
    example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels[i]]))
                    }))
    writer.write(record=example.SerializeToString())
writer.close()
