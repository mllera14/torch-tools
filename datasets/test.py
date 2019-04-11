import torch
from handwritten_numbers import load_hwnd
import matplotlib.pyplot as plt

########################################################################################
# LOAD DATA
########################################################################################

# Load data
data_path, size = '/home/milton/Code/Data/', 100

training, testing, _ = load_hwnd(data_path, size, 3, 50, val_split_ratio=0, download=False)

for numbers, labels in training:
    print(labels)
    for img in numbers:
        print(img.shape)
        plt.imshow(img.reshape(-1, 28), cmap='gray')
        plt.show()
        # for digit in img:
        #     plt.imshow(digit.reshape(28, 28 * 3), cmap='gray')
        #     plt.show()
