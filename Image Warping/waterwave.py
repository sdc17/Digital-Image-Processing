import numpy as np
import numpy.matlib
import random
from cv2 import cv2
from math import floor, ceil


source = 'media/ww.jpg'


def bilinear(x, y, source_img):
    f11 = source_img[floor(y), floor(x), :]
    f12 = source_img[ceil(y), floor(x), :]
    f21 = source_img[floor(y), ceil(x), :]
    f22 = source_img[floor(y), floor(x), :]
    mat1 = np.array([ceil(x) - x, x - floor(x)])
    mat2 = np.array([[f11, f12], [f21, f22]])
    mat3 = np.array([ceil(y) - y, y - floor(y)])
    if ceil(x) == floor(x) and ceil(y) == floor(y):
        f = f11
    elif ceil(x) == floor(x):
        f = f11*(ceil(y) - y)/(ceil(y)-floor(y)) + f12*(y-floor(y))/(ceil(y)-floor(y))
    elif ceil(y) == floor(y):
        f = f11*(ceil(x) - x)/(ceil(x)-floor(x)) + f21*(x-floor(x))/(ceil(x)-floor(x))
    else:
        f = np.array([
            mat1.dot(mat2[:, :, 0]).dot(mat3)/((ceil(x) - floor(x))*(ceil(y) - floor(y))),
            mat1.dot(mat2[:, :, 1]).dot(mat3)/((ceil(x) - floor(x))*(ceil(y) - floor(y))),
            mat1.dot(mat2[:, :, 2]).dot(mat3)/((ceil(x) - floor(x))*(ceil(y) - floor(y))),
        ])
    return f


def main(decay, velocity):
    source_img = cv2.imread(source)
    target_img = np.zeros_like(source_img)
    row, col= source_img.shape[0], source_img.shape[1]

    center = [col/2, row/2]

    x_rel = numpy.matlib.repmat(np.arange(col), row, 1) - center[0]
    y_rel = center[1] - np.transpose(numpy.matlib.repmat(np.arange(row), col, 1))

    theta = np.arctan2(y_rel, x_rel)
    r = np.sqrt(pow(x_rel, 2) + pow(y_rel, 2))+decay*col*np.sin(velocity*np.sqrt(pow(x_rel, 2)+pow(y_rel, 2)))

    x_out = r * np.cos(theta) + center[0]
    y_out = center[1] - r * np.sin(theta) 

    for i in range (row):
        for j in range (col):
            if x_out[i, j] >= 0 and x_out[i, j] <= col - 1 and y_out[i, j] >= 0 and y_out[i, j] <= row - 1:
                target_img[i, j, :] = bilinear(x_out[i, j], y_out[i, j], source_img)
            elif i == 0 and j == 0:
                target_img[i, j, :] = [0, 0, 0]
            elif i == 0:
                target_img[i, j, :] = target_img[i, j - 1, :]
            elif j == 0:
                target_img[i, j, :] = target_img[i - 1, j, :]
            else:
                if random.random() < 0.5:
                    target_img[i, j, :] = target_img[i, j - 1, :]
                else:
                    target_img[i, j, :] = target_img[i - 1, j, :]
                # target_img[i, j, :] = target_img[i - 1, j - 1, :]

    target_img = target_img.astype(np.uint8)
    cv2.namedWindow('Water wave', 0)
    cv2.moveWindow('Water wave', 480, 160)
    cv2.imwrite('results/wwtemp.png', target_img)
    cv2.imshow('Water wave', target_img)
    cv2.waitKey()
    cv2.destroyAllWindows()   


if __name__ == '__main__':
    main(0.04, 0.08)
