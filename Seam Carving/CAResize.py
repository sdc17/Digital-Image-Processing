import os
import cv2
import time
import math
import argparse
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from functools import reduce

l2 = lambda x: np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)


def generate_energy(input, mode='gradient'): # [gradient, HoG]
    input = cv2.GaussianBlur(input, (17, 17), 1)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    energy = reduce(lambda x, y: np.sqrt(x**2 + y**2), [nd.convolve(input[:, :, c], kernel, mode='nearest') for c in range(3)])
    if mode == 'HoG':
        HoG = cv2.HOGDescriptor()
        res = HoG.compute(input, (11, 11), (11, 11)).reshape((-1,))
        energy /= np.max(res)
    return abs(energy)

def remove_vertical_seam(input, tot, mode='backward'): # [backward, forward]
    height, width, channel = input.shape
    for k in range(tot):
        print('Seam {}/{}'.format(k, tot))     
        energy = generate_energy(input)

        dp = energy.copy()
        for j in range(1, height):
            dp[j][0] += min(dp[j - 1][0], dp[j - 1][1])
            for k in range(1, width - 1):
                if mode == 'backward':
                    dp[j][k] += min(dp[j - 1][k - 1], dp[j - 1][k], dp[j - 1][k + 1])
                else:
                    dp[j][k] += min(dp[j - 1][k - 1] + abs(input[j, k + 1, 0] - input[j, k - 1, 0]) + abs(input[j - 1, k, 0] - input[j, k - 1, 0]), \
                        dp[j - 1][k] + abs(input[j, k + 1, 0] - input[j, k - 1, 0]), \
                        dp[j - 1][k + 1] + abs(input[j - 1, k, 0] - input[j, k + 1, 0]))
            dp[j][width - 1] += min(dp[j - 1][width - 1], dp[j - 1][width - 2])
        
        trace = [np.argmin(dp[height - 1])]
        for i in range(height - 2, -1, -1):
            last = trace[-1]
            if last != 0 and last != width - 1:
                trace.append(last + np.argmin([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) - 1)
            elif last == 0:
                trace.append(last + np.argmin([dp[i][last], dp[i][last + 1]]))
            else:
                trace.append(last + np.argmin([dp[i][last - 1], dp[i][last]]) - 1)
        
        width -= 1

        for i in range(height):
            input[i, trace[height - 1 - i]:-1, :] = input[i, trace[height - 1 - i] + 1:, :]
        input = input[:, :-1, :]
        
    return input
    

def remove_horizontal_seam(input, tot):
    height, width, channel = input.shape
    for k in range(tot):
        print('Seam {}/{}'.format(k, tot))     
        energy = generate_energy(input)

        dp = energy.copy()
        for j in range(1, width):
            dp[0][j] += min(dp[0][j - 1], dp[1][j - 1])
            for k in range(1, height - 1):
                dp[k][j] += min(dp[k - 1][j - 1], dp[k][j - 1], dp[k + 1][j - 1])
            dp[height - 1][j] += min(dp[height - 1][j - 1], dp[height - 2][j - 1])

        trace = [np.argmin(dp[:, width - 1])]
        for i in range(width - 2, -1, -1):
            last = trace[-1]
            if last != 0 and last != height - 1:
                trace.append(last + np.argmin([dp[last - 1][i], dp[last][i], dp[last + 1][i]]) - 1)
            elif last == 0:
                trace.append(last + np.argmin([dp[last][i], dp[last + 1][i]]))
            else:
                trace.append(last + np.argmin([dp[last - 1][i], dp[last][i]]) - 1)
        
        height -= 1

        for i in range(width):
            input[trace[width - 1 - i]:-1, i, :] = input[trace[width - 1 - i] + 1:, i, :]
        input = input[:-1, :, :]
    
    return input


def insert_vertical_seam(input, tot, scale=3):
    tots = []
    scale -= 1
    for i in range(scale):
        tots.append(math.floor(tot/scale))
    if tot % scale != 0:
        tots.append(tot % scale)

    for cnt, tot in enumerate(tots):
        print('Iteration {}/{}'.format(cnt, len(tots)))
        height, width, channel = input.shape
        energy = generate_energy(input)

        dp = energy.copy()
        for j in range(1, height):
            dp[j][0] += min(dp[j - 1][0], dp[j - 1][1])
            for k in range(1, width - 1):
                dp[j][k] += min(dp[j - 1][k - 1], dp[j - 1][k], dp[j - 1][k + 1])
            dp[j][width - 1] += min(dp[j - 1][width - 1], dp[j - 1][width - 2])

        trace = np.zeros((tot, height)).astype(np.uint32)

        seam_to_ins = np.argsort(dp[height - 1])[:tot]
        occupied = 1e10
        for j in range(tot):
            trace[j][height - 1] = seam_to_ins[j]
            for i in range(height - 2, -1, -1):
                last = trace[j][i + 1]
                if last != 0 and last != width - 1:
                    # if (np.array([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) == occupied).all():
                    #     trace[j][i] = last
                    # else:
                    #     trace[j][i] = last + np.argmin([dp[i][last - 1], dp[i][last], dp[i][last + 1]]) - 1
                    left = 1
                    while dp[i][(last - left)%width] == occupied:
                        left += 1
                    right = 1
                    while dp[i][(last + right)%width] == occupied:
                        right += 1
                    choose = np.argmin([dp[i][(last - left)%width], dp[i][last], dp[i][(last + right)%width]])
                    if choose == 0:
                        trace[j][i] = (last - left)%width
                    elif choose == 1:
                        trace[j][i] = last
                    else:
                        trace[j][i] = (last + right)%width
                    dp[i][trace[j][i]] = occupied      
                elif last == 0:
                    # if (np.array([dp[i][last], dp[i][last + 1]]) == occupied).all():
                    #     trace[j][i] = last
                    # else:
                    #     trace[j][i] = last + np.argmin([dp[i][last], dp[i][last + 1]])
                    right = 1
                    while dp[i][(last + right)%width] == occupied:
                        right += 1
                    choose = np.argmin([dp[i][last], dp[i][(last + right)%width]])
                    if choose == 0:
                        trace[j][i] = last
                    else:
                        trace[j][i] = (last + right)%width
                    dp[i][trace[j][i]] = occupied
                else:
                    # if (np.array([dp[i][last - 1], dp[i][last]]) == occupied).all():
                    #     trace[j][i] = last
                    # else:
                    #     trace[j][i] = last + np.argmin([dp[i][last - 1], dp[i][last]]) - 1
                    left = 1
                    while dp[i][(last - left)%width] == occupied:
                        left += 1
                    choose = np.argmin([dp[i][(last - left)%width], dp[i][last]])
                    if choose == 0:
                        trace[j][i] = (last - left)%width
                    else:
                        trace[j][i] = last
                    dp[i][trace[j][i]] = occupied
                
        trace_tran = trace.transpose()

        output = np.zeros((height, width + tot, channel))
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
        
        input = output.copy()
    
    return input


def insert_horizontal_seam(input, tot):
    pass


def object_remove(input, label):
    height, width, channel = input.shape
    cnt = 0
    while True:
        find = False
        print('Removing seam {}'.format(cnt))
        for h in range(height):
            for w in range(width):
                if label[h][w] != 0:
                    energy = generate_energy(input)

                    dp = energy.copy()
                    for j in range(1, height):
                        dp[j][0] += min(dp[j - 1][0], dp[j - 1][1])
                        for k in range(1, width - 1):
                            dp[j][k] += min(dp[j - 1][k - 1], dp[j - 1][k], dp[j - 1][k + 1])
                        dp[j][width - 1] += min(dp[j - 1][width - 1], dp[j - 1][width - 2])

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

                    width -= 1

                    for i in range(height):
                        input[i, trace[i]:-1, :] = input[i, trace[i] + 1:, :]
                        label[i, trace[i]:-1] = label[i, trace[i] + 1:]
                    input, label = input[:, :-1, :], label[:, :-1]

                    find = True
                    break

            if find:
                break
        
        if not find:
            break
        
        cnt += 1
    
    return input

def main():
    parse = argparse.ArgumentParser(description='Point Processing')
    parse.add_argument('-i', dest='input', type=str, required=True, help='input file location')
    parse.add_argument('-o', dest='output', type=str, help='output file location')
    parse.add_argument('-th', dest='height', type=int, help='target height')
    parse.add_argument('-tw', dest='width', type=int, help='target width')
    parse.add_argument('-m', dest='mask', type=str, help='mask for object removal')
    args = parse.parse_args()

    assert os.path.exists(args.input)
    if args.height:
        assert args.height > 0
    if args.width:
        assert args.width > 0

    input = cv2.imread(args.input)
    height, width, _ = input.shape
    input_copy = input.copy()

    if args.mask:
        output = object_remove(input, np.load(args.mask))
    else:
        if args.height == height or args.width == width:
            output = input
        elif not args.height or args.height == height :
            if args.width < width:
                output = remove_vertical_seam(input, width - args.width)
            else:
                output = insert_vertical_seam(input, args.width - width)
        elif not args.width or args.width == width:
            if args.height < height:
                output = remove_horizontal_seam(input, height - args.height)
            else:
                output = insert_horizontal_seam(input, args.height - height)
        else:
            if args.width < width:
                output = remove_vertical_seam(input, width - args.width)
            else:
                output = insert_vertical_seam(input, args.width - width)
            if args.height < height:
                output = remove_horizontal_seam(input, height - args.height)
            else:
                output = insert_horizontal_seam(input, args.height - height)

    output_location = args.output if args.output else args.input.split('.')[0] + '_output.' + args.input.split('.')[-1]
    # output_location = args.output if args.output else args.input.split('.')[0] + '_hog_output.' + args.input.split('.')[-1]
    # output_location = args.output if args.output else args.input.split('.')[0] + '_forward_output.' + args.input.split('.')[-1]
    cv2.imwrite(output_location, output)

    if args.mask:
        print('{} object removal to {} done.'.format(args.input, output_location))
    else:
        print('{}({}) retargets to {}({}) done.'.format(input.shape, args.input, output.shape, output_location))

    
if __name__ == '__main__':
    main()