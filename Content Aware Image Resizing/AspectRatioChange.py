import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2


def main():
    parse = argparse.ArgumentParser(description='Point Processing')
    parse.add_argument('-i', dest='input', type=str, default='beach.jpg', help='input picture')
    parse.add_argument('-o', dest='output', type=str, default='output1.png', help='output picture')
    parse.add_argument('-r', dest='ratio', type=float, default = 1.4, help='new aspect ration')
    args = parse.parse_args()

    if args.input:
        input = cv2.imread(args.input)
        input_copy = input.copy()
        input_gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    else:
        print("Invalid input picture")
        return

    height, width, channel = input.shape
    origin_ratio = width/height
    if args.ratio <= 0 or args.ratio >= origin_ratio:
        print("Ration should between 0 and \\frac{width}{height}")
        return
    else:
        print("Input heigth: {} width: {}".format(height, width))

    # seam number
    tot = int(width - (height * args.ratio))
    mask = np.zeros_like(input_gray)
    
    # circle to remove seam besides the fisrt one
    for k in range(tot):
        # init energy
        energy = np.zeros((height, width))

        # corners
        energy[0][0] = abs(input_gray[1][0] - input_gray[0][0]) + abs(input_gray[0][1] - input_gray[0][0])
        energy[0][width - 1] = abs(input_gray[1][width - 1] - input_gray[0][width - 1]) + abs(input_gray[0][width - 2] - input_gray[0][width - 1])
        energy[height - 1][0] = abs(input_gray[height - 2][0] - input_gray[height - 1][0]) + abs(input_gray[height - 1][1] - input_gray[height - 1][0])
        energy[height - 1][width - 1] = abs(input_gray[height - 2][width - 1] - input_gray[height - 1][width - 1]) + \
            abs(input_gray[height - 1][width - 2] - input_gray[height - 1][width - 1])
        
        # boundary
        for i in range(1, height - 1):
            energy[i][0] = abs(input_gray[i + 1][0] - input_gray[i - 1][0])/2 + abs(input_gray[i][1] - input_gray[i][0])
            energy[i][width - 1] = abs(input_gray[i + 1][width - 1] - input_gray[i - 1][width - 1])/2 + abs(input_gray[i][width - 2] - input_gray[i][width - 1])
        for j in range(1, width - 1):
            energy[0][j] = abs(input_gray[0][j + 1] - input_gray[0][j - 1])/2 + abs(input_gray[1][j] - input_gray[0][j])
            energy[height - 1][j] = abs(input_gray[height - 1][j + 1] - input_gray[height - 1][j - 1])/2 + abs(input_gray[height - 2][j] - input_gray[height - 1][j])

        # inner
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                energy[i][j] = abs(input_gray[i + 1][j] - input_gray[i - 1][j])/2 + abs(input_gray[i][j + 1] - input_gray[i][j - 1])/2

        # init dp
        dp = np.zeros((height, width))
        for j in range(width):
            dp[0][j] = energy[0][j]
        for j in range(1, height):
            dp[j][0] = energy[j][0] + min(dp[j - 1][0], dp[j - 1][1])
            dp[j][width - 1] = energy[j][width - 1] + min(dp[j - 1][width - 1], dp[j - 1][width - 2])
            for k in range(1, width - 1):
                dp[j][k] = energy[j][k] + min(dp[j - 1][k - 1], dp[j - 1][k], dp[j - 1][k + 1])

        # backtrace
        trace = [np.argmin(dp[height - 1])]
        for i in range(height - 2, -1, -1):
            last = trace[-1]
            if last != 0 and last != width - 1:
                trace.append(last + np.argmin([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) - 1)
            elif last == 0:
                trace.append(last + np.argmin([dp[i][last], dp[i][last + 1]]))
            else:
                trace.append(last + np.argmin([dp[i][last - 1], dp[i][last]]) - 1)

        # update width
        width -= 1

        # update image
        new_input = np.zeros((height, width, channel)).astype(np.uint8)
        new_input_gray = np.zeros((height, width)).astype(np.uint8)
        for i in range(height):
            for j in range(width):
                if j < trace[height - 1 - i]:
                    new_input[i, j, :] = input[i, j, :]
                    new_input_gray[i][j] = input_gray[i][j]
                else:
                    new_input[i, j, :] = input[i, j + 1, :]
                    new_input_gray[i][j] = input_gray[i][j + 1]
        input = new_input
        input_gray = new_input_gray


    # show result
    # plt.imshow(input_gray, cmap='gray')
    plt.subplot(2, 1, 1)
    plt.imshow(input_copy[:, :, ::-1])
    plt.title("Origin image with width/height={:.2f}".format(origin_ratio))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 1, 2)
    plt.imshow(input[:, :, ::-1])
    plt.title("Output image with width/height={:.2f}".format(args.ratio))
    plt.xticks([])
    plt.yticks([])

    # plt.savefig(args.output)
    plt.show()


if __name__ == '__main__':
    main()
