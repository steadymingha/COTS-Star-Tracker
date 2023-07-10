import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.optimize import curve_fit
from skimage.measure import regionprops
import os
class Centroid:
    def __init__(self):
        pass

    def gaussian_2d(self, xy, amplitude, x0, y0, sigma_x, sigma_y):
        x, y = xy
        gaussian = amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))
        return gaussian.ravel()

    def Gaussian_fitting(self, image):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)

        ## initial guess
        # 이미지를 이진화합니다.
        image = np.clip(image, 20, np.max(image))
        image = (image - (np.ones((image.shape[0], image.shape[1])) * 20)).astype(np.uint8)

        _, bw = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

        # 레이블링을 수행합니다.
        ret, labels = cv2.connectedComponents(bw)

        # 각 객체의 속성을 계산합니다.
        props = regionprops(labels)

        # 각 객체의 중심, 주축 길이, 부축 길이를 구합니다.
        centers = np.array([prop.centroid for prop in props])
        major_axis_lengths = np.array([prop.major_axis_length for prop in props])
        minor_axis_lengths = np.array([prop.minor_axis_length for prop in props])

        # 각 객체의 지름과 반지름을 계산합니다.
        diameters = np.mean([major_axis_lengths, minor_axis_lengths], axis=0)
        radii = diameters / 2

        # 원을 그립니다.
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        for center, radius in zip(centers, radii):
            print("radius ", radius, center[1], center[0])

        # circle = plt.Circle((center[1], center[0]), radius, fill=False, color='r')
        # ax.add_patch(circle)
        # ax.plot(center[1], center[0], '.')
        # plt.show()

        psf_size = int((radius + 10) ** 2)

        # start_x = max(0, max_loc[0] - psf_size // 2)
        # end_x = min(image.shape[1], max_loc[0] + psf_size // 2)
        # start_y = max(0, max_loc[1] - psf_size // 2)
        # end_y = min(image.shape[0], max_loc[1] + psf_size // 2)

        start_x = max(0, int(center[1]) - (psf_size // 2))
        end_x = min(image.shape[1], int(center[1]) + (psf_size // 2))
        start_y = max(0, int(center[0]) - (psf_size // 2))
        end_y = min(image.shape[0], int(center[1]) + psf_size // 2)

        psf = image[(start_y):(end_y), (start_x):(end_x)]
        # psf = image[0:end_y, 0:end_x]

        x = np.arange(psf.shape[1])
        y = np.arange(psf.shape[0])
        x, y = np.meshgrid(x, y)

        psf_ravel = psf.ravel()

        initial_guess = (psf.max(), x.mean(), y.mean(), psf.shape[1] / 4, psf.shape[0] / 4)
        # initial_guess = (psf.max(), x.mean(), y.mean(), psf.shape[1], psf.shape[0])
        popt, _ = curve_fit(self.gaussian_2d, (x, y), psf_ravel, p0=initial_guess)

        amplitude, x0, y0, sigma_x, sigma_y = popt
        print('amplitude:', amplitude)
        print('x0:', x0)
        print('y0:', y0)
        print('sigma_x:', sigma_x)
        print('sigma_y:', sigma_y)

        return amplitude, sigma_x, sigma_y, (x0 + start_x), (y0 + start_y)


if __name__ == '__main__':

    # img_dir = 'ref_img/out000000000000000000000000000001.jpg.2022_1106_1715.51.848104.jpg'
    # img_dir = 'img_gen/test_r3.0_w0.05_d500.jpg'
    # img_dir = 'img_gen/radius_aperture/test_r10.0_w0.50_d1000.jpg'
    img_dir = 'img_gen/final_x260_y140.jpg'
    # img_dir = 'ref_img/out000000000000000000000000001002.jpg'
    # img_dir = 'ref_img/Focus0.jpg'
    # img_dir = 'ref_img/Focus2.jpg'


    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    print(os.path.basename(img_dir))

    # 미디언 필터 적용
    image = cv2.medianBlur(image, 5)



    cent = Centroid()
    amplitude, sigma_x, sigma_y, Gx, Gy = cent.Gaussian_fitting(image)


    # 원본 이미지에 광원의 중심 위치에 점을 찍습니다.
    # cv2.circle(image, (int(x0 + start_x), int(y0 + start_y)), radius=1, color=(255, 0, 255), thickness=50)
    plt.plot(int(Gx), int(Gy), 'r.')
    # 이미지를 출력합니다.
    print('final ','x','y',Gx, Gy)
    plt.imshow(image, cmap='gray')
    plt.show()
