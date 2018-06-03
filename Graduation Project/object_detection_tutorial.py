import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#capture a video
cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)
cap.set(cv2.CAP_PROP_EXPOSURE,50)

# What model to download.
MODEL_NAME = 'lhc_out_put2/'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'label_map.pbtxt')

NUM_CLASSES = 2 

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

#Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'IMG_{}.JPG'.format(i)) for i in range(1,5) ] 
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def return_image_loc():
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #for image_path in TEST_IMAGE_PATHS:
          #image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
        while(cap.isOpened()):
        #for i in range(1):
          ret, frame = cap.read()
          if ret == True:
              cv2.imshow('frame',frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              cv2.imwrite("/home/xz/testlll.jpg",frame)
              cap.release()
              cv2.destroyAllWindows()
        image_np = np.array(Image.open("/home/xz/testlll.jpg"))
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            #max_boxes_to_draw=2,
            use_normalized_coordinates=True,
            line_thickness=8)
        classes_detected = classes.astype('int')
        yida_index = np.where(classes_detected == 1)[1][0]
        #print(yida_index)
        #print("first yida's score:")
        #print(scores[0][yida_index])
        #print("yida's box location:")
        loc = boxes[0][yida_index]
        #print(loc)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.show()
    return loc
