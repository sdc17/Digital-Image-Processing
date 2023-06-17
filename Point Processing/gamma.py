import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import argparse


def gamma(input, value):
    return 255. * np.power(input / 255., value)


def get_hist(input):
    bins = np.arange(0, 256, 1.)
    hist, _ = np.histogram(input, bins=bins)
    return hist / input.size


def plot(pics, value, ope='Gamma'):
    plt.figure(figsize=(16, 10))
    for i, pic in enumerate(pics):
        img = cv2.imread(pic)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img[:, :, ::-1])
        if i == 0:
            title = ope + ' ^ ' + str(1 / value)
        elif i == 1:
            title = 'Origin Picture'
        else: 
            title = ope + ' ^ ' + str(value)
        plt.title(title, fontsize=12)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, i + 4)
        if i == 1:
            title = 'Origin'
        plt.title(title + ' pdf', fontsize=12)
        plt.plot(get_hist(img))
        plt.xlabel("Mean: {:.2f}  Std Dev: {:.2f}".format(np.mean(img), np.std(img)))

    plt.savefig('result_' + ope + '.jpg')
    plt.show()


def main():
    parse = argparse.ArgumentParser(description='Point Processing')
    parse.add_argument('-i', dest='input', type=str, default='input.jpg', help='input picture')
    parse.add_argument('-o', dest='output', type=str, default='output.jpg', help='output picture')
    parse.add_argument('-v', dest='value', type=float, default = 2., help='value for processing')
    args = parse.parse_args()

    if args.input:
        input = cv2.imread(args.input)
    else:
        print("Invalid input picture")
        return

    if args.value < 0.:
        print("Gamma value should be non-negative")
        return

    if args.value:
        output1 = args.output.split('.')[0] + '1.' + args.output.split('.')[1]
        cv2.imwrite(output1, gamma(input,  1. / args.value))
        output2 = args.output.split('.')[0] + '2.' + args.output.split('.')[1]
        cv2.imwrite(output2, gamma(input, args.value))
        plot([output1, args.input, output2], args.value)
        
    else:
        print("Invalid value for operation")
        return


if __name__ == '__main__':
    main()
