import os
import sys
import cv2
import ast
import shutil
import imageio
import argparse
import numpy as np
np.random.seed(114514)
import maxflow

l2 = lambda a, b: np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def cornerleft(hr, wr, visited_union):
    for i in range(hr - 1):
        for j in range(wr - 1):
            if visited_union[i, j] == 1 and visited_union[i + 1, j] == 1 and visited_union[i, j + 1] == 1 and visited_union[i + 1, j + 1] == 0:
                return i, j
    return -1, -1


def cornerright(hr, wr, visited_union):
    for i in range(hr - 1):
        for j in range(1, wr):
            if visited_union[i, j] == 1 and visited_union[i + 1, j] == 1 and visited_union[i, j - 1] == 1 and visited_union[i + 1, j - 1] == 0:
                return i, j
    return -1, -1

def overlapping(shape, visited, offset, cornerl=False, contain_point=None, windows_size=None, cornerr=False):
    height, width = shape
    x, y = offset
    hr, wr = visited.shape
    visited_current = np.zeros((hr, wr)).astype(np.uint8)
    if contain_point is None and windows_size is None:
        visited_current[x: x+ height, y: y + width] = 1
    else:
        containx, containy = contain_point
        wh, ww = windows_size
        x0, x1, y0, y1 = max(containx - wh//2, 0), min(containx + wh//2, hr-1) + 1, max(containy - ww//2, 0), min(containy + ww//2, wr-1) + 1
        visited_current[x0:x1, y0:y1] = 1
    visited_union = visited & visited_current
    flag = False
    for i in range(hr):
        for j in range(wr):
            if visited_union[i, j] == 1:
                top, left = i, j
                flag = True
                break
        if flag:
            break
    flag = False
    for i in reversed(range(hr)):
        for j in range(wr):
            if visited_union[i, j] == 1:
                bottom = i
                flag = True
                break
        if flag:
            break
    flag = False
    for i in range(hr):
        for j in reversed(range(wr)):
            if visited_union[i, j] == 1:
                right = j
                flag = True
                break
        if flag:
            break
    if cornerl and not cornerr:
        return top, left, bottom, right, visited_union, cornerleft(hr, wr, visited_union)
    
    if cornerl and cornerr:
        return top, left, bottom, right, visited_union, cornerleft(hr, wr, visited_union), cornerright(hr, wr, visited_union), (x0, x1, y0, y1)
    
    return top, left, bottom, right, visited_union
    

def entire_select(input, output, visited, min_val, max_val, constant=None):
    height, width, _ = input.shape
    var = np.var(input)
    costs = []
    
    marginal = round(height*0.1) if constant is None else round(width*0.1)
    for i in range(min_val+marginal, max_val-marginal, 1):
        if constant is None:
            top, left, bottom, right, visited_union = overlapping((height, width), visited, (i, 0))
        else:
            top, left, bottom, right, visited_union = overlapping((height, width), visited, (constant, i))

        tot = 0
        for j in range(top, bottom + 1, 1):
            for k in range(left, right + 1, 1):
                if visited_union[j, k] == 1:
                    tot += l2(output[j, k], input[j - top, k - left])
        ct = tot/np.sum(visited_union)
        costs.append(np.exp(-ct/(0.001*var)))
    costs /= np.sum(costs)
    return np.random.choice(list(range(min_val+marginal, max_val-marginal, 1)), p=np.array(costs))


def sub_select(input, output, visited, contain_point, windows_size):
    height, width, _  = input.shape
    var = np.var(input)
    hr, wr = visited.shape
    wh, ww = windows_size
    top, left, bottom, right, visited_union, (cornerlx, cornerly), (cornerrx, cornerry), limits = \
        overlapping((height, width), visited, (-1, -1), True, contain_point, windows_size, True)

    costs = []
    # for h in range(height-wh//2-1):
    #     for w in range(width-ww//2-1):
    for h in range(height-wh-1):
        for w in range(width-ww-1):
            ct = 0
            for j in range(top, bottom + 1, 1):
                for k in range(left, right + 1, 1):
                    if visited_union[j, k] == 1:
                        ct += l2(output[j, k], input[j - top + h, k - left + w])
            costs.append(np.exp(-ct/(var)))
    costs /= np.sum(costs)
    selected = np.random.choice(list(range(len(costs))), p=np.array(costs))
    # x, y = selected // (width-ww//2-1), selected % (width-ww//2-1)
    x, y = selected // (width-ww-1), selected % (width-ww-1)
    return x, y, top, left, bottom, right, visited_union, cornerlx, cornerly, cornerrx, cornerry, limits


def offset(input, output, visited, mode, windows_size=None):
    height, width, _ = input.shape
    len1, len2 = visited.shape
    flag = False
    for i in range(len1):
        for j in range(len2):
            if visited[i, j] == 0:
                containx, containy = i, j
                flag = True
                break
        if flag:
            break
    if not flag:
        return -1, -1
    min_x, max_x = max(containx - height + 1, 0), containx
    min_y, max_y = max(containy - width + 1, 0), containy

    if mode == 'random':
        if containy == 0:
            x = np.random.randint(min_x, max_x) if min_x != max_x else min_x
        else:
            for i in reversed(range(len1)):
                if visited[i, 0] == 1:
                    x = max(i - height + 1, 0)
                    break
        y = np.random.randint(min_y, max_y) if min_y != max_y else min_y

    elif mode == 'entire':
        if containy == 0:
            x = entire_select(input, output, visited, min_x, max_x) if min_x != max_x else min_x
        else:
            for i in reversed(range(len1)):
                if visited[i, 0] == 1:
                    x = max(i - height + 1, 0)
                    break
        y = entire_select(input, output, visited, min_y, max_y, x) if min_y != max_y else min_y

    else:
        return sub_select(input, output, visited, (containx, containy), windows_size)
    return x, y


def synthesis(input, results_dir, rate, mode, frames=True):
    filename = input.split('/')[-1].split('.')[0]
    print('Synthesis {} begin'.format(filename).center(60,"*"))
    input = imageio.mimread(input)[0]
    if args.input == './data/strawberries2.gif':
        input = cv2.cvtColor(np.array(input)[1:-1, 1:-1, :], cv2.COLOR_RGB2BGR)
    else:
        input = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)

    height, width, channel = input.shape
    
    # init output and visited
    hr, wr = round(height*rate[0]), round(width*rate[1])
    output = np.zeros((hr, wr, channel))
    visited = np.zeros((hr, wr)).astype(np.uint8)
    output[:height, :width, :] = input
    visited[:height, :width] = 1

    # iteration synthesis
    cnt = 0
    while True:
        if frames:
            cv2.imwrite(os.path.join(results_dir, filename + '_' + str(cnt) + '.png'), output)
        if mode in ['random', 'entire']:
            x, y = offset(input, output, visited, mode)
            if x == -1 and y == -1:
                break
            top, left, bottom, right, visited_union, (cornerx, cornery) = overlapping((height, width), visited, (x, y), True)
            visited[x: x+ height, y: y + width] = 1
        else:
            # windows_size = (2*height//3, 2*width//3)
            windows_size = (3*height//4, 3*width//4)
            returns = offset(input, output, visited, mode, windows_size)
            if returns[0] == -1 and returns[1] == -1:
                break
            x, y, top, left, bottom, right, visited_union, cornerlx, cornerly, cornerrx, cornerry, limits = returns
            visited[limits[0]: limits[1], limits[2]: limits[3]] = 1
        print('Interation {}: Offset({}, {})'.format(cnt, x, y))

        # from matplotlib import pyplot as ppl
        # ppl.imshow(visited_union)
        # ppl.show()
        
        g = maxflow.Graph[float]()
        h, w = max(bottom - top + 1, 0), max(right - left + 1, 0)

        nodeids = g.add_grid_nodes((h, w))
 
        structure = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, 0, 0]])
        weights = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w - 1):
                if mode in ['random', 'entire']:
                    weights[i, j] = l2(output[x + i, y + j, :], input[i, j, :]) + l2(output[x + i, y + j + 1, :], input[i, j + 1, :])
                else:
                    weights[i, j] = l2(output[top + i, left + j, :], input[x + i, y + j, :]) + l2(output[top + i, left + j + 1, :], input[x + i, y + j + 1, :])
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

        structure = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 1, 0]])
        weights = np.zeros((h, w))

        for i in range(h - 1):
            for j in range(w):
                if mode in ['random', 'entire']:
                    weights[i, j] = l2(output[x + i, y + j, :], input[i, j, :]) + l2(output[x + i + 1, y + j, :], input[i + 1, j, :])
                else:
                    weights[i, j] = l2(output[top + i, left + j, :], input[x + i, y + j, :]) + l2(output[top + i + 1, left + j, :], input[x + i + 1, y + j, :])
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

        if mode in ['random', 'entire']:
            if x == 0:
                g.add_grid_tedges(np.array([x[0] for x in nodeids]), float("inf"), float("-inf"))
                g.add_grid_tedges(np.array([x[-1] for x in nodeids]), float("-inf"), float("inf"))
            elif y == 0:
                g.add_grid_tedges(nodeids[0], float("inf"), float("-inf"))
                g.add_grid_tedges(nodeids[-1], float("-inf"), float("inf"))
            else:
                g.add_grid_tedges(np.append(np.array([x[0] for x in nodeids]), nodeids[0, 1:]), float("inf"), float("-inf"))
                g.add_grid_tedges(nodeids[cornerx - top:, cornery - left:], float("-inf"), float("inf"))
        else:
            if top == 0:
                g.add_grid_tedges(np.array([x[0] for x in nodeids]), float("inf"), float("-inf"))
                g.add_grid_tedges(np.array([x[-1] for x in nodeids]), float("-inf"), float("inf"))
            elif left == 0:
                g.add_grid_tedges(nodeids[0], float("inf"), float("-inf"))
                g.add_grid_tedges(nodeids[-1], float("-inf"), float("inf"))
            elif cornerlx != -1 and cornerly != -1 and cornerrx != -1 and cornerry != -1:
                g.add_grid_tedges(np.array([x[0] for x in nodeids] + list(nodeids[0, 1:-1]) + [x[-1] for x in nodeids]), float("inf"), float("-inf"))
                g.add_grid_tedges(nodeids[cornerlx - top:, cornerly - left:cornerry - left + 1], float("-inf"), float("inf"))
            elif cornerlx != -1 and cornerly != -1:
                g.add_grid_tedges(np.append(np.array([x[0] for x in nodeids]), nodeids[0, 1:]), float("inf"), float("-inf"))
                g.add_grid_tedges(nodeids[cornerlx - top:, cornerly - left:], float("-inf"), float("inf"))
            elif cornerrx != -1 and cornerry != -1:
                g.add_grid_tedges(np.append(np.array([x[-1] for x in nodeids]), nodeids[0, :-1]), float("inf"), float("-inf"))
                g.add_grid_tedges(nodeids[cornerrx - top:, :cornerry - left + 1], float("-inf"), float("inf"))

        g.maxflow()
        sgm = g.get_grid_segments(nodeids)

        # from matplotlib import pyplot as ppl
        # ppl.imshow(np.int_(np.logical_not(sgm)))
        # ppl.show()

        if mode in ['random', 'entire']:
            for i in range(height):
                for j in range(width):
                    if not(i < h and j < w and not(sgm[i, j])) and i + x < hr and j + y < wr:
                        output[i + x, j + y, :] = input[i, j, :]
        else:
            for i in range(limits[0], limits[1]):
                for j in range(limits[2], limits[3]):
                    if not(i - limits[0] < h and j - limits[2]< w and not(sgm[i - limits[0], j - limits[2]])):
                        output[i, j, :] = input[x + i - limits[0], y + j - limits[2], :]
        cnt += 1

    save_path = os.path.join(results_dir, filename + '.png')
    cv2.imwrite(save_path, output)
    print('Result location:{}'.format(save_path))
    print('Synthesis end'.format(filename).center(60,"*"))


if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='Textures Synthesis')
    parse.add_argument('-i', dest='input', type=str, default='./data/green.gif', help='input file location')
    parse.add_argument('-m', dest='mode', type=str, default = 'random', help='offset mode')
    parse.add_argument('-f', dest='frames', default=True, type=ast.literal_eval, choices=[True, False], help="Output intraprocess frames")
    args = parse.parse_args()

    if not args.input :
        raise AttributeError("Invalid input file location")
        
    if args.mode not in ['random', 'entire', 'sub']:
        raise AttributeError('Supported modes are random, entire or submatch.')

    results_dir = 'results_' + args.input.split('/')[-1].split('.')[0] + '_' + args.mode
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    synthesis(args.input, results_dir, (2, 2), args.mode, args.frames)