# Import Statements
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Reading the Image(Converting image to 3D array)
sample_image=  cv2.imread(r"C:\Users\harsh\OneDrive\Desktop\neural arrt task.jpeg") # Reads an image into BGR Format

# BGR to RGB conversion
sample_image = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

# Converting 3D array of image to 2D array
all_pixels  = sample_image.reshape((-1,3))

# Specify "k" ie. the no. of dominant colours to be present in the final image
k = 45

# KMeans Algorithm
km = KMeans(n_clusters=k)
km.fit(all_pixels)

# Fetching the centroids in a list in RGB format
centers = km.cluster_centers_

new_img = np.zeros((all_pixels.shape),dtype='uint8')

# Iterating over the image
for ix in range(new_img.shape[0]):
    new_img[ix] = centers[km.labels_[ix]]

# Displaying the final image
new_img = new_img.reshape((sample_image.shape))
plt.imshow(new_img)
plt.axis("off")
plt.show()