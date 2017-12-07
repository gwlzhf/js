import tensorflow as tf
from PIL import Image, ImageDraw
import dataset_util

tf_record_filename_queue = tf.train.string_input_producer(["m_locust_test.record"])
# print(tf_record_filename_queue)
tf_record_reader = tf.TFRecordReader()
_, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)
# print(tf_record_serialized)
# print(tf.Session().run(aaa,feed_dict={aaa:aaa}))
# 通过阅读器读取value值,并保存为tf_record_serialized

# The label and image are stored as bytes but could be stored as int64 or float64 values in a
# serialized tf.Example protobuf.
# 标签和图像都按字节存储,但也可按int64或float64类型存储于序列化的tf.Example protobuf文件中
tf_record_features = tf.parse_single_example(tf_record_serialized,features={
	'image/height': tf.VarLenFeature(tf.int64),
	'image/width': tf.VarLenFeature(tf.int64),
	'image/filename': tf.VarLenFeature(tf.string),
	'image/source_id': tf.VarLenFeature(tf.string),
	'image/encoded': tf.VarLenFeature(tf.string),
	'image/format': tf.VarLenFeature(tf.string),
	'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
	'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
	'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
	'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),


	# 'image/object/bbox/ymax': tf.FixedLenFeature([2],dtype=tf.float32),

	'image/object/class/text': tf.VarLenFeature(tf.string),
	'image/object/class/label': tf.VarLenFeature(tf.int64)
	})
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())


with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	# features = sess.run(tf_record_features)
	# print(features)
	height = tf_record_features['image/height']
	width = tf_record_features['image/width']
	filename = tf_record_features['image/filename']
	xmins = tf_record_features['image/object/bbox/xmin']
	xmaxs = tf_record_features['image/object/bbox/xmax']
	ymins = tf_record_features['image/object/bbox/ymin']
	ymaxs = tf_record_features['image/object/bbox/ymax']
	[height,width,filename,xmins,xmaxs,ymins,ymaxs] = sess.run([height,width,filename,xmins,xmaxs,ymins,ymaxs])
	height = tf_record_features['image/height']
	width = tf_record_features['image/width']
	filename = tf_record_features['image/filename']
	xmins = tf_record_features['image/object/bbox/xmin']
	xmaxs = tf_record_features['image/object/bbox/xmax']
	ymins = tf_record_features['image/object/bbox/ymin']
	ymaxs = tf_record_features['image/object/bbox/ymax']
	[height,width,filename,xmins,xmaxs,ymins,ymaxs] = sess.run([height,width,filename,xmins,xmaxs,ymins,ymaxs])
	height = height.values
	width = width.values
	filename = filename.values
	xmins = xmins.values
	ymins = ymins.values
	xmaxs = xmaxs.values
	ymaxs = ymaxs.values
	img = Image.open(filename[0])
	drawsurface = ImageDraw.Draw(img)

	for i in range(len(xmins)):
		drawsurface.rectangle([xmins[i]*width, ymins[i]*height, xmaxs[i]*width, ymaxs[i]*height])
	img.show()
	print(filename[0])



	# print(ymaxs.values)
	# print(filename)



# im = Image.open('m_locust 0.JPG')
# drawsurface = ImageDraw.Draw(im)
# drawsurface.rectangle([50,100,100,200])
# im.show()
