import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import argparse


def get_cdf(input):
    bins = np.arange(0, 256, 1.)
    hist, _ = np.histogram(input, bins=bins)
    hist = np.insert(hist, 0, 0)
    return np.cumsum(hist / hist.sum())


def get_hist(input):
    bins = np.arange(0, 256, 1.)
    hist, _ = np.histogram(input, bins=bins)
    return hist / input.size


def LUT(input, target):
    input_cdf = get_cdf(input)
    target_cdf = get_cdf(target)
    cdf_map = np.full(256, 255)
    for i in range(255):
        for j in range(256):
            if input_cdf[i] < target_cdf[j]:
                cdf_map[i] = j
                break
    return cdf_map[input]


def matching(input, target):
    return np.dstack((LUT(input[:, :, 0], target[:, :, 0]), LUT(input[:, :, 1], target[:, :, 1]),
    LUT(input[:, :, 2], target[:, :, 2])))


def plot(pics, ope='Histogram Matching'):
    plt.figure(figsize=(16, 10))
    for i, pic in enumerate(pics):
        img = cv2.imread(pic)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img[:, :, ::-1])
        if i == 0:
            title = 'Origin Picture'
        elif i == 1:
            title = 'Target Picture'
        else: 
            title = 'Remapped Picture'
        plt.title(title, fontsize=12)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, i + 4)
        plt.title(title + ' pdf', fontsize=12)
        plt.plot(get_hist(img))
        plt.xlabel("Mean: {:.2f}  Std Dev: {:.2f}".format(np.mean(img), np.std(img)))

    plt.savefig('result_' + ope + '.jpg')
    plt.show()


def main():
    parse = argparse.ArgumentParser(description='Point Processing')
    parse.add_argument('-i', dest='input', type=str, default='input.jpg', help='input picture')
    parse.add_argument('-o', dest='output', type=str, default='output.jpg', help='output picture')
    parse.add_argument('-v', dest='value', type=str, default = 'input2.jpg', help='target picture')
    args = parse.parse_args()

    if args.input:
        input = cv2.imread(args.input)
    else:
        print("Invalid input picture")
        return

    if args.value:
        target = cv2.imread(args.value)
    else:
        print("Invalid target picture")
        return

    cv2.imwrite(args.output, matching(input, target))
    plot([args.input, args.value, args.output])


def mulit():
    input = cv2.imread('input.jpg')
    target1 = cv2.imread('input4.jpg')
    target2 = cv2.imread('input3.jpg')
    target3 = cv2.imread('input5.jpg')
    cv2.imwrite('output1.jpg', matching(input, target1))
    cv2.imwrite('output2.jpg', matching(input, target2))
    cv2.imwrite('output3.jpg', matching(input, target3))

    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(input[:, :, ::-1])
    plt.title('Origin Picture', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    
    for i, item in enumerate([target1, target2, target3]):
        plt.subplot(3, 4, 3 + 4 * i)
        plt.imshow(item[:, :, ::-1])
        if i == 0:
            plt.title('Target Picture', fontsize=12)
        plt.xticks([])
        plt.yticks([])


    for i, item in enumerate(['output1.jpg', 'output2.jpg', 'output3.jpg']):
        img = cv2.imread(item)
        plt.subplot(3, 4, 4 + 4 * i)
        plt.imshow(img[:, :, ::-1])
        if i == 0:
            plt.title('Remapped Picture', fontsize=12)
        plt.xticks([])
        plt.yticks([])

    plt.savefig('result_Histogram Matching_mul.jpg')
    plt.show()


if __name__ == '__main__':
    main()
    # mulit()
