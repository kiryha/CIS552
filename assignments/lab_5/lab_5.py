import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


def task_1():

    # In cv2, a RGB image is read as a BGR array
    img_bgr = cv2.imread('images_src/Island.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(img_bgr)
    plt.title('img_bgr')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title('img_rgb')

    # In cv2, a BGR array is written as a RGB image
    cv2.imwrite('images_out/img_bgr.png', img_bgr)
    cv2.imwrite('images_out/img_rgb.png', img_rgb)


def task_2():

    img_bgr = cv2.imread('images_src/Island.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_bgr_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    print(img_bgr.shape)
    print(img_rgb_gray.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(img_bgr_gray, cmap='gray')
    plt.title('img_bgr_gray')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb_gray, cmap='gray')
    plt.title('img_rgb_gray')

    cv2.imwrite('images_out/img_bgr_gray.png', img_bgr_gray)
    cv2.imwrite('images_out/img_rgb_gray.png', img_rgb_gray)

    t = cv2.imread('images_out/img_bgr_gray.png')
    print(t.shape)
    s = cv2.imread('images_out/img_bgr_gray.png', 0)
    print(s.shape)

    cv2.imshow('gray', img_bgr_gray)
    cv2.waitKey(0)


def task_3():

    img_bgr = cv2.imread('images_src/Lenna.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img3a = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    img3b = cv2.rotate(img_rgb, cv2.ROTATE_180)
    img3c = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img3d = cv2.resize(img_rgb, (600, 480))

    plt.figure(figsize=(12, 12))  # size in inch
    plt.subplot(1, 4, 1)
    plt.imshow(img3a)
    plt.title('ROTATE_90_CLOCKWISE')

    plt.subplot(1, 4, 2)
    plt.imshow(img3b)
    plt.title('ROTATE_180')

    plt.subplot(1, 4, 3)
    plt.imshow(img3c)
    plt.title('ROTATE_90_CLOCKWISE')

    plt.subplot(1, 4, 4)
    plt.imshow(img3d)
    plt.title('Resize to 600x480')
    plt.show()


def task_4():

    img_bgr = cv2.imread('images_src/Lenna.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Crop an image
    sizex, sizey, _ = img_rgb.shape
    img4a = img_rgb[int(sizex * 0.25):int(sizex * 0.75), int(sizey * 0.25):int(sizey * 0.75)]

    plt.imshow(img4a)
    plt.title('CROP')
    plt.show()


def task_5():

    img_bgr = cv2.imread('images_src/Lenna.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Rotation at any angle
    img5a = imutils.rotate(img_rgb, 32)  # 32 degree
    plt.imshow(img5a)
    plt.title('ROTATE')
    plt.show()


task_5()
