import numpy as np
from cv2 import cv2
from math import sin, cos, asin, atan, pi, sqrt, floor, ceil
import matplotlib.pyplot as plt


source = 'media/sphere.jpg'


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


def main():
    source_img = cv2.imread(source)

    target_size = min(np.array(source_img.shape)[:2])
    rho0 = target_size / 2
    d0 = max(np.array(source_img.shape)[:2]) / 2

    source_heigth = source_img.shape[0]
    source_width = source_img.shape[1]
    source_center = [source_heigth / 2, source_width / 2]

    target_img = np.zeros((target_size, target_size, 3))
    for i in range(target_size):
        for j in range(target_size):
            distance_square = pow(i - rho0, 2) + pow(j - rho0, 2)
            if(distance_square <= pow(rho0, 2)):
                rho = sqrt(distance_square)
                theta = atan((i - rho0)/(j - rho0 + 1e-8))
                phi = asin(rho/rho0)
                d = (2 * d0 * phi) / pi
                r_in = d * sin(theta)
                c_in = d * cos(theta)

                if j - rho0 < 0:
                    # i_in, j_in = round(source_center[0] - r_in), round(source_center[1] - c_in)
                    i_in, j_in = source_center[0] - r_in, source_center[1] - c_in
                    if(i_in < 0 or i_in > source_heigth - 1 or j_in < 0 or j_in > source_width - 1):
                        target_img[i, j, :] = [128, 128, 128]
                        continue
                    else:
                        # target_img[i, j, :] = source_img[i_in.astype(np.int), j_in.astype(np.int), :] # without bilinear
                        target_img[i, j, :] = bilinear(j_in, i_in, source_img)
                else:
                    # i_in, j_in = round(source_center[0] + r_in), round(source_center[1] + c_in)
                    i_in, j_in = source_center[0] + r_in, source_center[1] + c_in
                    if(i_in < 0 or i_in > source_heigth -1 or j_in < 0 or j_in > source_width - 1):
                        target_img[i, j, :] = [128, 128, 128]
                        continue
                    else:
                        # target_img[i, j, :] = source_img[i_in.astype(np.int), j_in.astype(np.int), :] # without bilinear
                        target_img[i, j, :] = bilinear(j_in, i_in, source_img)
    
    target_img = target_img.astype(np.uint8)
    cv2.imwrite('results/sphere.png', target_img)
    cv2.imshow('Sphere Warping', target_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
