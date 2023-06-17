import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import linalg, signal


def main(input, target, mask, paste_x, paste_y):
    
    input, target, mask = cv2.imread(args.input), cv2.imread(args.target), cv2.imread(args.mask)
    height, width, channel = input.shape
    
    # compute divergence
    kernel = np.asarray([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
    input_div = np.dstack([signal.convolve2d(input[:, :, c], kernel) for c in range(3)])
    target_div = np.dstack([signal.convolve2d(target[:, :, c], kernel) for c in range(3)])
    
    mask[mask > 0] = 255
    input_mask = input & mask
    is_mask = lambda x, y : mask[x, y, 0] != 0
    is_counter = lambda x, y : is_mask(x, y) and (not is_mask(x, y - 1) or not is_mask(x, y + 1) or not is_mask(x - 1, y) or not is_mask(x + 1, y))
    
    # number mask points
    number = np.zeros_like(mask[:, :, 0]).astype(np.uint32)
    cnt = 0
    for i in range(number.shape[0]):
        for j in range(number.shape[1]):
            if is_mask(i, j):
                number[i, j] = cnt
                cnt += 1

    N =  np.count_nonzero(mask[:, :, 0])
    A = np.zeros((N, N))
    b = np.zeros((N, 3))

    # find basic position
    base_x, base_y, cnt = -1, -1, 0
    for i in range(height):
        for j in range(width):
            if is_mask(i, j):
                base_y = i
                break
        if base_y != -1:
            break
    
    for j in range(width):
        for i in range(height):
            if is_mask(i, j):
                base_x = j
                break
        if base_x != -1:
            break

    # construct matrix A and vector b
    for i in range(height):
        for j in range(width):
            if is_counter(i, j):
                A[cnt, cnt] = 1
                b[cnt, :] = target[i - base_y + paste_y, j - base_x + paste_x, :]
                cnt += 1
            elif is_mask(i, j):
                A[cnt, cnt] = -4
                A[cnt, number[i - 1, j]] = A[cnt, number[i + 1, j]] = A[cnt, number[i, j - 1]] = A[cnt, number[i, j + 1]] = 1
                b[cnt, :] = input_div[i, j, :]
                cnt += 1

    # solve Ax=b
    x = np.clip(np.stack([linalg.solve(A, b[:, c]) for c in range(3)], axis=1), 0, 255).astype(np.uint8)
    
    # plot origin image
    plt.subplot(1, 2, 1)
    plt.imshow(target[:, :, ::-1])
    plt.title('Origin Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    # update image
    for i in range(number.shape[0]):
        for j in range(number.shape[1]):
            if number[i, j] != 0:
                target[i - base_y + paste_y, j - base_x + paste_x, :] = x[number[i, j]]

    # plot image after fusion
    plt.subplot(1, 2, 2)
    plt.imshow(target[:, :, ::-1])
    plt.title('After Fusion', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Image Fusion')

    # test1
    parse.add_argument('-i', dest='input', type=str, default='./images/part1/test1_src.jpg', help='input image file location')
    parse.add_argument('-t', dest='target', type=str, default = './images/part1/test1_target.jpg', help='input target file location')
    parse.add_argument('-m', dest='mask', type=str, default = './images/part1/test1_mask.jpg', help='input mask file location')
    parse.add_argument('-x', dest='pastex', type=int, default = 50, help='where to paste x axes')
    parse.add_argument('-y', dest='pastey', type=int, default = 100, help='where to paste y axes')

    # test2
    # parse.add_argument('-i', dest='input', type=str, default='./images/part1/test2_src.png', help='input image file location')
    # parse.add_argument('-t', dest='target', type=str, default = './images/part1/test2_target.png', help='input target file location')
    # parse.add_argument('-m', dest='mask', type=str, default = './images/part1/test2_mask.png', help='input mask file location')
    # parse.add_argument('-x', dest='pastex', type=int, default = 150, help='where to paste x axes')
    # parse.add_argument('-y', dest='pastey', type=int, default = 180, help='where to paste y axes')
    

    args = parse.parse_args()

    if not args.input or not args.target or not args.mask:
        print("Invalid parameters")
        exit(-1)

    main(args.input, args.target, args.mask, args.pastex, args.pastey)
