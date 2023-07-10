import os
import numpy as np
import matplotlib.pyplot as plt


def draw_ellipse(size=(500, 500), center=None, savedir=None, radius=(100, 50)):
    # Initialize the image with black color
    image = np.zeros(size)

    # If center is not provided, then we put the circle in the middle of the image
    if center is None:
        center = (size[0] // 2, size[1] // 2)

    # Generate two 2-D arrays representing the X and Y coordinates in the image
    y, x = np.ogrid[:size[0], :size[1]]


    # Equation of ellipse: (x-center_x)^2/radius_x^2 + (y-center_y)^2/radius_y^2 = 1
    # dist_from_center = np.sqrt((x - center[0]) ** 2 / radius[0] ** 2 + (y - center[1]) ** 2 / radius[1] ** 2)
    dist_from_center = np.sqrt(np.power((x - center[0]),2, dtype=np.double) / np.power(radius[0],2, dtype=np.double) + np.power((y - center[1]),2, dtype=np.double) / np.power(radius[1],2, dtype=np.double))

    # Creating the ellipse with gradient from white at center to black at boundary
    # We utilize the "1 - dist_from_center" part to achieve the gradient effect
    mask = np.clip(1 - dist_from_center, 0, 1)

    # Add the ellipse to the existing image
    image = np.maximum(image, mask)

    # Show the image
    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    plt.imsave(os.path.join(img_save_dir, 'ellipse_%f_%f.png' % (center[0], center[1])), image, cmap='gray')


def draw_circle(size, center=None, savedir=None, radius=100, sigma=0):
    image = np.zeros(size)


    # if center is None:
    #     center = (size[0] // 2, size[1] // 2)

    y, x = np.ogrid[:size[0], :size[1]]

    # Equation of circle: (x-center_x)^2 + (y-center_y)^2 = radius^2
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Creating the circle with gradient from white at center to black at boundary
    # We utilize the "radius - dist_from_center" part to achieve the gradient effect
    # mask = np.clip((radius - dist_from_center) / radius, 0, 1)
    # mask = np.clip((radius - dist_from_center ** 2) / radius, 0, 1)

    # Gaussian function
    # sigma = radius  # Standard deviation
    mask = 2* np.exp(-dist_from_center ** 2 / (2 * sigma ** 2))

    # Add the circle to the existing image
    image = np.maximum(image, mask)




    # Show the image
    plt.imshow(image, cmap='gray')



    plt.imsave(os.path.join(img_save_dir, 'circle_%f_%f_%d.png' % (center[0],center[1],sigma)),image,cmap='gray')


    # plt.axis('off')
    # plt.tight_layout(pad=0)
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # # plt.show()
    # plt.savefig(os.path.join(img_save_dir, 'circle_%f_%f.png' % (center[0],center[1])), transparent = True, bbox_inches = 'tight', pad_inches = 0)



if __name__== "__main__":
    img_save_dir = '/home/user/Work/COTS-Star-Tracker/test/test_single'
    # draw_circle((320, 240), center=(59.23453, 35.9876), savedir=img_save_dir, radius=20)
    # draw_circle((240, 320), center=(59.23453, 35.9876), savedir=img_save_dir, radius=20)
    draw_circle((240, 320), center=(98.1235, 87.324), savedir=img_save_dir, radius=20,sigma=0)
    # draw_ellipse((240, 320), center=(89.32, 124.2356), savedir=img_save_dir,radius=(46.342, 32))
    # draw_ellipse((240, 320), center=(102.32, 124.2356), savedir=img_save_dir,radius=(45.23, 31.4))