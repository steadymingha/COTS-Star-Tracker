
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math



# cv2 blur types
# https://076923.github.io/posts/C-opencv4-14/
x_center = 301.4533456
y_center = 254.56753
img_save_dir = './'
background = np.zeros((600,500), dtype=np.uint8)

mask = np.zeros((255,255), np.uint8)

# xx and yy are 200x200 tables containing the x and y coordinates as values
# mgrid is a mesh creation helper
xx, yy = np.mgrid[:600, :500]
# circles contains the squared distance to the (100, 100) point
# we are just using the circle equation learnt at school
# circle = (xx - 100) ** 2 + (yy - 100) ** 2
circle = np.power(xx-x_center,2, dtype=np.double) + np.power(yy-y_center,2, dtype=np.double)


# donuts contains 1's and 0's organized in a donut shape
# you apply 2 thresholds on circle to define the shape
# donut = np.logical_and(circle < (6400 + 60), circle > (6400 - 60))


fig, ax = plt.subplots()
ax.set_xlim(-1000, 1000)
ax.set_ylim(-300, 1000)
rect = plt.Rectangle((0, 0), 10000, 10000, color='black')
# ax.add_patch(rect)


# plt.imshow(circle, cmap='gray')
# plt.imshow(circle, cmap='binary')

plt.imshow(circle, cmap='gist_yarg')
# plt.show()
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(os.path.join(img_save_dir,'handmade_plt.png'), bbox_inches = 'tight', pad_inches = 0)
# test = 0



# # using CV2 -> center point should be integer
# # mask = cv2.circle(background, (250, 250), 30, (255,255,255), -1, cv2.LINE_AA)
# mask = cv2.GaussianBlur(mask, (17,17),7 )
# # dst = cv2.blur(src, ksize, dst, anchor, borderType)
# # mask = cv2.medianBlur(mask, 99)

# plt.imshow(mask, cmap='gray')
# plt.show()
# cv2.imwrite(os.path.join(img_save_dir,'handmade.png'),mask)





# # https://stackoverflow.com/questions/56130052/how-can-i-apply-a-gaussian-blur-to-a-figure-in-matplotlib
# https://stackoverflow.com/questions/10031580/how-to-write-simple-geometric-shapes-into-numpy-arrays








