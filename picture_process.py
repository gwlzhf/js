import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('../Picture/l.png','rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_png(image_raw_data)
    print(image_data.eval)

    plt.imshow(img_data.eval())
    plt.show()

    
