import cv2

import tkinter
import matplotlib
matplotlib.use( 'tkagg' )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np

from map_image import MapImage

image_id = '0af5086ade'
image_path = 'data/train/0af5086ade.png'
# image_path=map_id_to_path[image_id]
map_image = MapImage(image_id=image_id, image_path=image_path)
# map_image.plot()

map_image.resize_by_scaling_factor(0.25)
# map_image.plot()

# r, g, b = cv2.split(map_image.image)
# fig = plt.figure(figsize=(20,20))
# axis = fig.add_subplot(1, 1, 1, projection="3d")

img = map_image.image
pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# RGB parameterization
rgb_img = img
r, g, b = cv2.split(img)

fig = plt.figure(figsize=(20, 10))

# axis = fig.add_subplot(1, 1, 1, projection="3d")
axis = fig.add_subplot(1, 2, 1, projection='3d')
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

axis = fig.add_subplot(1, 2, 2, projection='3d')
axis.scatter(g.flatten(), r.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Green")
axis.set_ylabel("Red")
axis.set_zlabel("Blue")

plt.show()

# HSV parameterization
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)

fig = plt.figure(figsize=(20, 10))

axis = fig.add_subplot(1, 2, 1, projection='3d')
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")

axis = fig.add_subplot(1, 2, 2, projection='3d')
axis.scatter(s.flatten(), h.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Saturation")
axis.set_ylabel("Hue")
axis.set_zlabel("Value")

plt.show()