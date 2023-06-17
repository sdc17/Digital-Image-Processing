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


def LUT(input):
    return 255*get_cdf(input)[input]


def equalization(input):
    return np.dstack((LUT(input[:, :, 0]), LUT(input[:, :, 1]), LUT(input[:, :, 2])))


def plot(pics, ope='Histogram Equalization'):
    plt.figure(figsize=(16, 10))
    for i, pic in enumerate(pics):
        img = cv2.imread(pic)
        plt.subplot(2, 2, i * 2 + 1)
        plt.imshow(img[:, :, ::-1])
        if i == 0:
            title = 'Origin Picture'
        else: 
            title = ope
        plt.title(title, fontsize=12)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 2, i * 2 + 2)
        if i == 0:
            plt.title('Corresponding pdf', fontsize=12)
        plt.xlabel("Mean: {:.2f}  Std Dev: {:.2f}".format(np.mean(img), np.std(img)))
        plt.plot(get_hist(img))


    plt.savefig('result_' + ope + '.jpg')
    plt.show()


def main():
    parse = argparse.ArgumentParser(description='Point Processing')
    parse.add_argument('-i', dest='input', type=str, default='input.jpg', help='input picture')
    parse.add_argument('-o', dest='output', type=str, default='output.jpg', help='output picture')
    args = parse.parse_args()

    if args.input:
        input = cv2.imread(args.input)
    else:
        print("Invalid input picture")
        return

    cv2.imwrite(args.output, equalization(input))
    plot([args.input, args.output])


if __name__ == '__main__':
    main()
