# Import Statements
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# Reading the Image(Converting image to 3D array)
sample_image=  cv2.imread(r"C:\Users\harsh\OneDrive\Desktop\neural arrt task.jpeg") # Reads an image into BGR Format

# BGR to RGB conversion
sample_image = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

# Converting 3D array of image to 2D array
all_pixels  = sample_image.reshape((-1,3))

# Specify "k" ie. the no. of dominant colours to be present in the final image
k = 4

# KMeans Algorithm
centers={}  # Stores the Centroids
for i in range(k):
    centers[i]=[]
for i in range(k):
    centers[i]=random.choice(all_pixels)

for i in range(10):
    labels=[] # Same as labels in Sklearn
    clusters={}
    for j in range(k):
                clusters[j] = []
    for pixel in all_pixels:
        distances = [np.linalg.norm(pixel-centers[center]) for center in centers]
        label = distances.index(min(distances))
        labels.append(label)
        cluster = distances.index(min(distances))
        clusters[cluster].append(pixel)
    for cluster in clusters:
                centers[cluster] = np.average(clusters[cluster], axis=0)

new_img = np.zeros((all_pixels.shape), dtype='uint8')

# Iterate over the image
for i in range(new_img.shape[0]):
    new_img[i] = centers[labels[i]]

# Coverting the 2D array back to 3D array
new_img = new_img.reshape((sample_image.shape))

# Displaying the final image
plt.imshow(new_img)
plt.axis("off")
plt.show()
