import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2


def main():
    parse = argparse.ArgumentParser(description='Point Processing')
    parse.add_argument('-i', dest='input', type=str, default='beach.jpg', help='input picture')
    parse.add_argument('-o', dest='output', type=str, default='output2.png', help='output picture')
    args = parse.parse_args()

    if args.input:
        input = cv2.imread(args.input)
        input_copy = input.copy()
        input_gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        input_vis = cv2.imread('./seg_beach/SegmentationClassVisualization/beach.jpg')
    else:
        print("Invalid input picture")
        return

    height, width, channel = input.shape

    label = np.load('./seg_beach/SegmentationClass/beach.npy')
    
    # remove object
    while True:
        find = False
        for h in range(height):
            for w in range(width):
                if label[h][w] != 0:
                    # init energy
                    energy = np.zeros((height, width))

                    # corners
                    energy[0][0] = abs(input_gray[1][0] - input_gray[0][0]) + abs(input_gray[0][1] - input_gray[0][0])
                    energy[0][width - 1] = abs(input_gray[1][width - 1] - input_gray[0][width - 1]) + abs(input_gray[0][width - 2] - input_gray[0][width - 1])
                    energy[height - 1][0] = abs(input_gray[height - 2][0] - input_gray[height - 1][0]) + abs(input_gray[height - 1][1] - input_gray[height - 1][0])
                    energy[height - 1][width - 1] = abs(input_gray[height - 2][width - 1] - input_gray[height - 1][width - 1]) + \
                        abs(input_gray[height - 1][width - 2] - input_gray[height - 1][width - 1])
                    
                    # rectangle
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
                    trace = np.zeros((height)).astype(np.int32)
                    trace[h] = w
                    for i in range(h - 1, -1, -1):
                        last = trace[i + 1]
                        if last != 0 and last != width - 1:
                            trace[i] = last + np.argmin([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) - 1
                        elif last == 0:
                            trace[i] = last + np.argmin([dp[i][last], dp[i][last + 1]])
                        else:
                            trace[i] = last + np.argmin([dp[i][last - 1], dp[i][last]]) - 1
                    for i in range(h + 1, height):
                        last = trace[i - 1]
                        if last != 0 and last != width - 1:
                            trace[i] = last + np.argmin([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) - 1
                        elif last == 0:
                            trace[i] = last + np.argmin([dp[i][last], dp[i][last + 1]])
                        else:
                            trace[i] = last + np.argmin([dp[i][last - 1], dp[i][last]]) - 1
                    
                    # update width
                    width -= 1

                    # update image
                    new_input = np.zeros((height, width, channel)).astype(np.uint8)
                    new_input_gray = np.zeros((height, width)).astype(np.uint8)
                    new_label = np.zeros((height, width)).astype(np.uint8)
                    for i in range(height):
                        for j in range(width):
                            if j < trace[i]:
                                new_input[i, j, :] = input[i, j, :]
                                new_input_gray[i][j] = input_gray[i][j]
                                new_label[i][j] = label[i][j]
                            else:
                                new_input[i, j, :] = input[i, j + 1, :]
                                new_input_gray[i][j] = input_gray[i][j + 1]
                                new_label[i][j] = label[i][j + 1]
                    input = new_input
                    input_gray = new_input_gray
                    label = new_label

                    # break
                    find = True
                    break

            if find:
                break
        
        if not find:
            break
    
    # seam to insertion
    tot = input_copy.shape[1] - width

    # init energy
    energy = np.zeros((height, width))

    # corners
    energy[0][0] = abs(input_gray[1][0] - input_gray[0][0]) + abs(input_gray[0][1] - input_gray[0][0])
    energy[0][width - 1] = abs(input_gray[1][width - 1] - input_gray[0][width - 1]) + abs(input_gray[0][width - 2] - input_gray[0][width - 1])
    energy[height - 1][0] = abs(input_gray[height - 2][0] - input_gray[height - 1][0]) + abs(input_gray[height - 1][1] - input_gray[height - 1][0])
    energy[height - 1][width - 1] = abs(input_gray[height - 2][width - 1] - input_gray[height - 1][width - 1]) + \
    abs(input_gray[height - 1][width - 2] - input_gray[height - 1][width - 1])
                    
    # rectangle
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
    trace = np.zeros((tot, height)).astype(np.uint32)

    # seam_to_ins_tot = np.argsort(dp[height - 1])[:(tot*5)]
    # seam_to_ins = np.zeros((tot))
    # for i in range(tot):
    #     seam_to_ins[i] = seam_to_ins_tot[(i*5)]

    seam_to_ins_tot = np.argsort(dp[height - 1])
    seam_to_ins = np.zeros((tot))
    pos = 0
    for i in range(tot):
        while abs(seam_to_ins_tot[pos+1] - seam_to_ins_tot[pos]) <= 2 * i:
            pos += 1
        seam_to_ins[i] = seam_to_ins_tot[pos]
        pos += 1

    # seam_to_ins_tot = np.array([]).astype(np.uint32)
    # seam_to_ins = np.array([]).astype(np.uint32)
    # for i in range(29):
    #     seam_to_ins_tot = np.append(seam_to_ins_tot, np.argsort(dp[height - 1][int(i*width/30):int((i+1)*width/30)])[:int(tot/30)*1] + int(i*width/30))
    # for i in range(len(seam_to_ins_tot)):
    #     if i % 1 == 0:
    #         seam_to_ins = np.append(seam_to_ins, seam_to_ins_tot[i])
    # seam_last = np.argsort(dp[height - 1][int(29*width/30):])
    # for i in range(len(seam_last)):
    #     if len(seam_to_ins) < tot and i % 1 == 0:
    #         seam_to_ins = np.append(seam_to_ins, seam_last[i] + int(29*width/30))

    for j in range(tot):
        trace[j][height - 1] = seam_to_ins[j]
        for i in range(height - 2, -1, -1):
            last = trace[j][i + 1]
            if last != 0 and last != width - 1:
                trace[j][i] = last + np.argmin([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) - 1
            elif last == 0:
                trace[j][i] = last + np.argmin([dp[i][last], dp[i][last + 1]])
            else:
                trace[j][i] = last + np.argmin([dp[i][last - 1], dp[i][last]]) - 1
    trace_tran = trace.transpose()

    # enlarging 
    output = np.zeros_like(input_copy)
    for i in range(height):
        pos = 0
        fills = np.sort(trace_tran[i])
        for j in range(width):
            if pos >= tot or j != fills[pos]:
                output[i, j + pos, :] = input[i, j, :]
            else:
                while pos < tot and j == fills[pos]:
                    output[i, j + pos, :] = input[i, j, :]
                    output[i, j + pos + 1, :] = input[i, j, :]
                    pos += 1
    
    # show result
    # plt.imshow(input_gray, cmap='gray')
    plt.subplot(2, 2, 1)
    plt.imshow(input_copy[:, :, ::-1])
    plt.title("Origin image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(input_vis[:, :, ::-1])
    plt.title("Object to remove")
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 3)
    plt.imshow(input[:, :, ::-1])
    plt.title("Image after removal")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.imshow(output[:, :, ::-1])
    plt.title("Image after removal and enlarging")
    plt.xticks([])
    plt.yticks([])

    # plt.savefig(args.output)
    plt.show()


if __name__ == '__main__':
    main()
