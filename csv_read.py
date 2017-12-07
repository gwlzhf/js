import pandas as pd
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt


test_data = pd.read_csv("test.csv")
pixel = np.matrix(test_data.iloc[10:,0:],dtype=np.uint8)

for i in range(10):
    image_matrix = np.reshape(pixel[i],[28,28])
    image = Image.fromarray(image_matrix)
    plt.imshow(image,cmap=plt.cm.gray,interpolation='nearest')
    plt.show()



