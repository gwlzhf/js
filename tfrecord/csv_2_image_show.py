import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
train_labels = train_data.pop(item="label")
train_values = train_data.values

for i in range(10):
    print(train_labels[i])
    plt.imshow(np.reshape(train_data.values[i],[28,28]))
    plt.show()
