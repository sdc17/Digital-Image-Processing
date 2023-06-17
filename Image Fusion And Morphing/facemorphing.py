import ast
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import linalg
from math import sin, cos, asin, atan, pi, sqrt, floor, ceil


def delaunay(fpath, show=False):
    json_raw = {}
    points = []
    with open(fpath.replace('.png', '.json').replace('.jpg', '.json'), 'r') as f:
        json_raw = json.load(f)
    for item in json_raw['shapes']:
        points.append((round(item['points'][0][0]), round(item['points'][0][1])))
        # points.append(tuple(item['points'][0]))
    img = cv2.imread(fpath)
    height, width, channel = img.shape
    
    # add points on boundary
    # 94 totally for test1
    points.extend([(0, 0), (round(width/2), 0), (round(width-1), 0),
                   (0, round(height/2)), (round(width/2), round(height/2)), (round(width-1), round(height/2)),
                   (0, round(height-1)), (round(width/2), round(height-1)), (round(width-1), round(height-1)),
    ])
    # points.extend([(0, 0), (width/2, 0), (width-1, 0),
    #                (0, height/2), (width/2, height/2), (width-1, height/2),
    #                (0, height-1), (width/2, height-1), (width-1, height-1),
    # ])

    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    for point in points:
        subdiv.insert(point)
    trai_list = subdiv.getTriangleList()

    ########################################################################################

    is_in_rect = lambda rect, point : point[0] >= rect[0] and point[0] <= rect[2] \
            and point[1] >= rect[1] and point[1] <= rect[3]

    trai_list_to_delete = []
    for i in range(len(trai_list)):
        if not is_in_rect(rect, (trai_list[i][0], trai_list[i][1])) or not is_in_rect(rect,
         (trai_list[i][2], trai_list[i][3])) or not is_in_rect(rect, (trai_list[i][4], trai_list[i][5])):
            trai_list_to_delete.append(i)
    trai_list = np.delete(trai_list, trai_list_to_delete, axis=0)

    ########################################################################################
    
    if show:
        # draw delaunay
        for i, trai in enumerate(trai_list):
            p1 = (trai[0], trai[1])
            p2 = (trai[2], trai[3])
            p3 = (trai[4], trai[5])
            if is_in_rect(rect, p1) and is_in_rect(rect, p2) and is_in_rect(rect, p3):
                cv2.line(img, p1, p2, (255, 245, 0), 1, cv2.LINE_AA, 0)
                cv2.line(img, p2, p3, (255, 245, 0), 1, cv2.LINE_AA, 0)
                cv2.line(img, p3, p1, (255, 245, 0), 1, cv2.LINE_AA, 0)
            
        # draw points
        draw_points = lambda img, p, color: cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)

        for point in points:
            draw_points(img, (int(point[0]), int(point[1])), (0, 0, 255))

        cv2.imshow("Delaunay", img)
        cv2.waitKey(0)

    return points, trai_list.astype(np.uint32)


def is_inner_trai(trai, p):
    a = 1e-30 + (-trai[1][1]*trai[2][0] + trai[0][1]*(-trai[1][0] + trai[2][0]) + trai[0][0]*(trai[1][1] - trai[2][1]) + trai[1][0]*trai[2][1])/2
    s = 1/(2*a)*(trai[0][1]*trai[2][0] - trai[0][0]*trai[2][1] + (trai[2][1] - trai[0][1])*p[0] + (trai[0][0] - trai[2][0])*p[1])
    t = 1/(2*a)*(trai[0][0]*trai[1][1] - trai[0][1]*trai[1][0] + (trai[0][1] - trai[1][1])*p[0] + (trai[1][0] - trai[0][0])*p[1])
    # return s > 0 and t > 0 and s + t < 1 # for excluding pixels on boundary
    return s >= 0 and t >= 0 and s + t <= 1 # pixels on boundary are included


def get_bounding_box(trai):
    xmin = min([x[0] for x in trai])
    ymin = min([x[1] for x in trai])
    xmax = max([x[0] for x in trai])
    ymax = max([x[1] for x in trai])
    return (xmin, ymin, xmax, ymax)


def bilinear(x, y, source_img):
    # overflow case
    if ceil(x) >= source_img.shape[1] or ceil(y) >= source_img.shape[0]:
        return source_img[int(y), int(x), :]
    # normal case
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


def solve_affine(t1, t2):
    A = np.zeros((6, 6))
    b = np.zeros(6)
    A[0, 0], A[0, 1], A[0, 2] = t1[0][0], t1[0][1], 1
    A[1, 3], A[1, 4], A[1, 5] = t1[0][0], t1[0][1], 1
    A[2, 0], A[2, 1], A[2, 2] = t1[1][0], t1[1][1], 1
    A[3, 3], A[3, 4], A[3, 5] = t1[1][0], t1[1][1], 1
    A[4, 0], A[4, 1], A[4, 2] = t1[2][0], t1[2][1], 1
    A[5, 3], A[5, 4], A[5, 5] = t1[2][0], t1[2][1], 1
    b[0], b[1] = t2[0][0], t2[0][1]
    b[2], b[3] = t2[1][0], t2[1][1]
    b[4], b[5] = t2[2][0], t2[2][1]
    x = linalg.solve(A, b)
    x = np.append(x, [0, 0, 1]).reshape(3, 3)
    # return x # for forward warping
    return linalg.inv(x) # for inverse warping


def apply_affine(input, target, res, t1, t2, t, t1_to_t, t2_to_t, alpha, use_bilinear=True):
    bb = get_bounding_box(t)
    for i in range(bb[1], bb[3] + 1):
        for j in range(bb[0], bb[2] + 1):
            if is_inner_trai(t, (j, i)):

                from_t1_pos = np.dot(t1_to_t, np.asarray([j, i, 1]))
                from_t1_pos /= from_t1_pos[2]
                from_t2_pos = np.dot(t2_to_t, np.asarray([j, i, 1]))
                from_t2_pos /= from_t2_pos[2]

                if use_bilinear: # with more time cost but higher accuracy
                    from_input = bilinear(from_t1_pos[0], from_t1_pos[1], input)
                    from_target = bilinear(from_t2_pos[0], from_t2_pos[1], target)
                else: # with less time cose but lower accuracy
                    from_input = input[int(from_t1_pos[1]), int(from_t1_pos[0]), :]
                    from_target = target[int(from_t2_pos[1]), int(from_t2_pos[0]), :]

                res[i, j, :] = (1 - alpha) * from_input + alpha * from_target
           
    return res

def morphing(input, points_input, trai_input, target, points_target, trai_target, alpha=0.5, use_bilinear=True):
    
    input, target = cv2.imread(args.input), cv2.imread(args.target)
    height = max(input.shape[0], target.shape[0])
    weight = max(input.shape[1], target.shape[1])

    res = np.zeros((height, weight, 3))

    points = []
    for i in range(len(points_input)):
        x = (1 - alpha) * points_input[i][0] + alpha * points_target[i][0]
        y = (1 - alpha) * points_input[i][1] + alpha * points_target[i][1]
        points.append((round(x), round(y)))

    for item in trai_input:
        v1 = points_input.index((item[0], item[1]))
        v2 = points_input.index((item[2], item[3]))
        v3 = points_input.index((item[4], item[5]))

        t1 = [points_input[v1], points_input[v2], points_input[v3]]
        t2 = [points_target[v1], points_target[v2], points_target[v3]]
        t = [points[v1], points[v2], points[v3]]
        
        t1_to_t = solve_affine(t1, t)
        t2_to_t = solve_affine(t2, t)

        res = apply_affine(input, target, res, t1, t2, t, t1_to_t, t2_to_t, alpha, use_bilinear)
        
    res = res.astype(np.uint8)
    return res


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Face Morphing')
    parse.add_argument('-i', dest='input', type=str, default='./images/part2-1/source1.png', help='input image file location')
    parse.add_argument('-t', dest='target', type=str, default = './images/part2-1/target1.png', help='input target file location')
    parse.add_argument('-s', dest='sequence', type=int, default = 5, help='length of the morphing sequence')
    parse.add_argument('-b', dest='bilinear', default=True, type=ast.literal_eval, choices=[True, False], help='whether to use bilinear')
    args = parse.parse_args()

    if not args.input or not args.target:
        print("Invalid parameters")
        exit(-1)

    if not isinstance(args.sequence, int) or args.sequence <= 0:
        print("Sequence should be an positive integr")
        exit(-1)

    # delaunay triangulation
    points_input, trai_input = delaunay(args.input, show=False)
    points_target, trai_target = delaunay(args.target, show=False)

    # plot input image
    plt.figure(figsize=(20, 12))
    plt.subplot(1, args.sequence + 2, 1)
    plt.imshow(cv2.imread(args.input)[:, :, ::-1])
    plt.title('Origin Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    # plot morphing sequences
    for i in range(args.sequence):
        alpha = (i + 1) / (args.sequence + 1)
        res = morphing(args.input, points_input,trai_input, args.target, points_target, trai_target, alpha)
        plt.subplot(1, args.sequence + 2, i + 2)
        plt.imshow(res[:, :, ::-1])
        plt.title('Morphing \\alpha={:.2f}'.format(alpha), fontsize=12)
        plt.xticks([])
        plt.yticks([])

    # plot target image
    plt.subplot(1, args.sequence + 2, args.sequence + 2)
    plt.imshow(cv2.imread(args.target)[:, :, ::-1])
    plt.title('Target Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()
