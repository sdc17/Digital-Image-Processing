import ast
import json
import dlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import linalg, optimize
from math import sin, cos, asin, atan, pi, sqrt, floor, ceil


def read_feature_points(fpath):
    json_raw = {}
    points = []
    with open(fpath.replace('.png', '.json').replace('.jpg', '.json'), 'r') as f:
        json_raw = json.load(f)
    for item in json_raw['shapes']:
        points.append([round(item['points'][0][0]), round(item['points'][0][1])])
    return points


def normalize(points):
    # redirect the centroid of points to origin
    mean_xy = np.mean(points, axis=0)
    mean_points = points - mean_xy

    # transform distance form origin to \sqrt{2}
    scale = np.sqrt(2) / np.mean(np.sqrt(mean_points[:, 0]**2 + mean_points[:, 1] **2))
    t_norm = np.asarray([[scale, 0, -scale*mean_xy[0]],
                         [0, scale, -scale * mean_xy[1]],
                         [0, 0, 1]
    ])
    homogenous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    points_normalized = np.dot(t_norm, homogenous.T).T
    
    # return transformed points in the form of homogenous and corresponding transformation matrix
    return points_normalized, t_norm


def compute_fundamental_matrix(input, target, t_i, t_t):
    # A equation matrix (x'x, x'y, x', y'x, y'y, y', x, y, 1) shape=(n, 9)
    A = np.concatenate((
        np.multiply(target[:, 0], input[:, 0]), np.multiply(target[:, 0], input[:, 1]), target[:, 0],
        np.multiply(target[:, 1], input[:, 0]), np.multiply(target[:, 1], input[:, 1]), target[:, 1],
        # input[:, 0], input[:, 1], np.ones((len(input), 1))[:, 0]
        input[:, 0], input[:, 1], np.ones(len(input))
    )).reshape((len(input), -1), order='F')

    # SVD for A
    U, D, V = np.linalg.svd(A)

    # extract smallest singular value, i.e., the last row of V
    F = V[-1, :].T.reshape(3, 3)

    # singularity constraint: F to be singular and of rank 2
    U, D, V = np.linalg.svd(F)

    # argmin(||F-F^'||_{Frobenius})
    F = np.dot(U, np.diag([D[0], D[1], 0])).dot(V)

    # denormalize
    F = t_t.T.dot(F).dot(t_i)

    return  F


def compute_rotation(v, t):
    # get rotation matrix
    ct = np.cos(t)
    st = np.sin(t)
    ro = np.array([[(1-ct)*v[0]*v[0] + ct, (1-ct)*v[0]*v[1], st*v[1]],
                     [(1-ct)*v[0]*v[1], (1-ct)*v[1]*v[1] + ct, -st*v[0]],
                     [-st*v[1], st*v[0], ct]])
    return ro


def compute_prewarp(F):
    # get eigenvalues and eigenvectors
    val0, vec0 = np.linalg.eig(F)
    val1, vec1 = np.linalg.eig(F.T)

    # get epipoles
    e0, e1 = vec0[:, np.argmin(val0)], vec1[:, np.argmin(val1)]

    # get rotation axis
    d0 = np.array([-e0[1], e0[0], 0])
    
    # get corresponding axis in input
    F0 = F.dot(d0)
    d1 = np.array([-F0[1], F0[0], 0])

    # get rotation angle
    t0, t1 = np.arctan(e0[2]/(d0[1]*e0[0] - d0[0]*e0[1])), np.arctan(e1[2]/(d1[1]*e1[0] - d1[0]*e1[1]))
    
    # get raotaiton matrix
    rdt0, rdt1 = compute_rotation(d0, t0), compute_rotation(d1, t1)
    
    # update epipoles
    re0, re1 = rdt0.dot(e0), rdt1.dot(e1)

    # update angle
    p0, p1 = -np.arctan(re0[1]/re0[0]), -np.arctan(re1[1]/re1[0])

    # rotation given by p0 and p1
    rp0 = np.array([[np.cos(p0), -np.sin(p0), 0],
                    [np.sin(p0), np.cos(p0), 0],
                    [0, 0, 1]])
    rp1 = np.array([[np.cos(p1), -np.sin(p1), 0],
                    [np.sin(p1), np.cos(p1), 0],
                    [0, 0, 1]])

    H0, H1 = rp0.dot(rdt0), rp1.dot(rdt1)

    return H0, H1


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


def solve_prjective(t1, t2):
    projective = lambda x : np.asarray([
        t2[0][0]-(x[0]*t1[0][0]+x[1]*t1[0][1]+x[2]*1)/(x[6]*t1[0][0]+x[7]*t1[0][1]+1*1),
        t2[0][1]-(x[3]*t1[0][0]+x[4]*t1[0][1]+x[5]*1)/(x[6]*t1[0][0]+x[7]*t1[0][1]+1*1),
        t2[1][0]-(x[0]*t1[1][0]+x[1]*t1[1][1]+x[2]*1)/(x[6]*t1[1][0]+x[7]*t1[1][1]+1*1),
        t2[1][1]-(x[3]*t1[1][0]+x[4]*t1[1][1]+x[5]*1)/(x[6]*t1[1][0]+x[7]*t1[1][1]+1*1),
        t2[2][0]-(x[0]*t1[2][0]+x[1]*t1[2][1]+x[2]*1)/(x[6]*t1[2][0]+x[7]*t1[2][1]+1*1),
        t2[2][1]-(x[3]*t1[2][0]+x[4]*t1[2][1]+x[5]*1)/(x[6]*t1[2][0]+x[7]*t1[2][1]+1*1),
        t2[3][0]-(x[0]*t1[3][0]+x[1]*t1[3][1]+x[2]*1)/(x[6]*t1[3][0]+x[7]*t1[3][1]+1*1),
        t2[3][1]-(x[3]*t1[3][0]+x[4]*t1[3][1]+x[5]*1)/(x[6]*t1[3][0]+x[7]*t1[3][1]+1*1)
    ])

    co = optimize.fsolve(projective, np.zeros(8))
    co = np.append(co, 1).reshape(3, 3)
    return co # for forward warping
    # return linalg.inv(co) # for inverse warping


def do_projective_warping(img, co, h, w, corners=False, shiftx=5, shifty=200): # test1
# def do_projective_warping(img, co, h, w, corners=False, shiftx=300, shifty=200): # test2 BAD DON'T USE
# def do_projective_warping(img, co, h, w, corners=False, shiftx=100, shifty=100): # test3
    res = np.zeros((h + shifty, w + shiftx, 3)).astype(np.uint8)
    co_inv = np.linalg.inv(co)
    for i in range(h + shifty):
        for j in range(w + shiftx):
            from_pos = np.dot(co_inv, np.array([j - shiftx, i - shifty, 1]))
            from_pos /= from_pos[2]
            if from_pos[0] >= 0 and from_pos[0] < img.shape[1] and from_pos[1] >= 0 and from_pos[1] < img.shape[0]:
                res[i, j, :] = bilinear(from_pos[0], from_pos[1], img)
    
    if not corners:
        return res, []

    #######################################  BUNK  ############################################
    #   I don't want to explain why I have written these tedious codes to compute             #
    #   some not really important corenrs, beacause I have been writing these homwork for a   #
    #   full two days.                                                                        #
    #   It always comes out some incomprehensible problems here or there but I have to move on#
    #   beacause there is no time left.                                                       #
    #   Writing these useless codes actually makes me feel at ease to some extent.            #
    ###########################################################################################

    is_valid = lambda pos : pos[0] + shiftx >= 0 and pos[0] < w  and pos[1] + shifty >= 0 and pos[1] < h 
    cnt = 1
    left_up = np.dot(co, np.array([0, 0, 1]))
    left_up /= left_up[2]
    while not is_valid(left_up):
        left_up = np.dot(co, np.array([0 + cnt, 0 + cnt - 1, 1]))
        left_up /= left_up[2]
        if is_valid(left_up):
            break
        left_up = np.dot(co, np.array([0 + cnt - 1, 0 + cnt, 1]))
        left_up /= left_up[2]
        if is_valid(left_up):
            break
        cnt += 1
    cnt = 1
    right_up  = np.dot(co, np.array([0, img.shape[0] - 1, 1]))
    right_up /= right_up[2]
    while not is_valid(right_up):
        right_up  = np.dot(co, np.array([0 + cnt, img.shape[0] - 1 - (cnt - 1), 1]))
        right_up /= right_up[2]
        if is_valid(right_up):
            break
        right_up  = np.dot(co, np.array([0 + cnt - 1, img.shape[0] - 1 - cnt, 1]))
        right_up /= right_up[2]
        if is_valid(right_up):
            break
        cnt += 1
    cnt = 1
    left_bottom = np.dot(co, np.array([img.shape[1] - 1, 0, 1]))
    left_bottom /= left_bottom[2]
    while not is_valid(left_bottom):
        left_bottom = np.dot(co, np.array([img.shape[1] - 1 - cnt, 0 + cnt - 1, 1]))
        left_bottom /= left_bottom[2]
        if is_valid(left_bottom):
            break
        left_bottom = np.dot(co, np.array([img.shape[1] - 1 - (cnt - 1), 0 + cnt, 1]))
        left_bottom /= left_bottom[2]
        if is_valid(left_bottom):
            break
        cnt += 1
    cnt = 1
    right_bottom = np.dot(co, np.array([img.shape[1] - 1, img.shape[0] - 1, 1]))
    right_bottom /= right_bottom[2]
    while not is_valid(right_bottom):
        right_bottom = np.dot(co, np.array([img.shape[1] - 1 - cnt, img.shape[0] - 1 - (cnt - 1), 1]))
        right_bottom /= right_bottom[2]
        if is_valid(right_bottom):
            break
        right_bottom = np.dot(co, np.array([img.shape[1] - 1 - (cnt - 1), img.shape[0] - 1 - cnt, 1]))
        right_bottom /= right_bottom[2]
        if is_valid(right_bottom):
            break
        cnt += 1
    # corners = [tuple(left_up[0:2].astype(np.uint32)), tuple(right_up[0:2].astype(np.uint32)), tuple(left_bottom[0:2].astype(np.uint32)), tuple(right_bottom[0:2].astype(np.uint32))]
    corners = [
        (int(left_up[0] + shiftx), int(left_up[1] + shifty)),
        (int(right_up[0] + shiftx), int(right_up[1] + shifty)),
        (int(left_bottom[0] + shiftx), int(left_bottom[1] + shifty)),
        (int(right_bottom[0] + shiftx), int(right_bottom[1] + shifty))
    ]
    corners.extend([
        (int((corners[0][0] + corners[1][0])/2), int((corners[0][1] + corners[1][1])/2)),
        (int((corners[0][0] + corners[2][0])/2), int((corners[0][1] + corners[2][1])/2)),
        (int((corners[3][0] + corners[1][0])/2), int((corners[3][1] + corners[1][1])/2)),
        (int((corners[3][0] + corners[2][0])/2), int((corners[3][1] + corners[2][1])/2)),
    ])
    return res, corners


def feature_points_detection(img, show=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./detection/shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    points = []
    key_points_set = [37, 40, 43, 46, 31, 49, 55, 58]
    img_copy = img.copy()
    for det in dets:
        shape = predictor(gray, det)
        for i, p in enumerate(shape.parts()):
            points.append((p.x, p.y))
            # if show and ((i + 1) in key_points_set):
            if show:
                cv2.circle(img_copy, (p.x, p.y), 2, (0, 0, 255), 1)
    if show:
        cv2.imshow("feature points", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points
    

def delaunay(img, points, corners=None, show=False):
    
    height, width, channel = img.shape
    
    # add points on boundary
    points.extend(corners)

    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    for point in points:
        subdiv.insert(point)
    trai_list = subdiv.getTriangleList()
    
    is_in_rect = lambda rect, point : point[0] >= rect[0] and point[0] <= rect[2] \
            and point[1] >= rect[1] and point[1] <= rect[3]

    trai_list_to_delete = []
    for i in range(len(trai_list)):
        if not is_in_rect(rect, (trai_list[i][0], trai_list[i][1])) or not is_in_rect(rect,
         (trai_list[i][2], trai_list[i][3])) or not is_in_rect(rect, (trai_list[i][4], trai_list[i][5])):
            trai_list_to_delete.append(i)
    trai_list = np.delete(trai_list, trai_list_to_delete, axis=0)

    if show:
        # draw delaunay
        img_copy = img.copy()
        for i, trai in enumerate(trai_list):
            p1 = (trai[0], trai[1])
            p2 = (trai[2], trai[3])
            p3 = (trai[4], trai[5])
            if is_in_rect(rect, p1) and is_in_rect(rect, p2) and is_in_rect(rect, p3):
                cv2.line(img_copy, p1, p2, (255, 245, 0), 1, cv2.LINE_AA, 0)
                cv2.line(img_copy, p2, p3, (255, 245, 0), 1, cv2.LINE_AA, 0)
                cv2.line(img_copy, p3, p1, (255, 245, 0), 1, cv2.LINE_AA, 0)
            
        # draw points
        draw_points = lambda img, p, color: cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)

        for point in points:
            draw_points(img_copy, (int(point[0]), int(point[1])), (0, 0, 255))

        cv2.imshow("Delaunay", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    #######################################  WARNING  ################################
    #   Some value of sequence length can lead to a magic alpha between input and    #
    #   target, which will raise LinAlgError for singualar matrix.                   #  
    #   I don't know how to fix it beacuse I am not good at linear algebra,          #
    #   so if you don't want to receive a crash, just take 5 as the value of         #
    #   sequence lenghth instead of other unsafe integers.                           #
    ##################################################################################
    parse = argparse.ArgumentParser(description='Face Morphing')
    parse.add_argument('-i', dest='input', type=str, default='./images/part2-2/source_1.png', help='input image file location')
    parse.add_argument('-t', dest='target', type=str, default = './images/part2-2/target_1.png', help='input target file location')
    parse.add_argument('-s', dest='sequence', type=int, default = 5, help='length of the morphing sequence')
    parse.add_argument('-b', dest='bilinear', default=True, type=ast.literal_eval, choices=[True, False], help='whether to use bilinear')
    parse.add_argument('-a', dest='auto', default=False, type=ast.literal_eval, choices=[True, False], help='use feature points label or dlib automatical detection')
    parse.add_argument('-x', dest='shiftx', type=int, default = 5, help='shift x pixels for image after prewarping')
    parse.add_argument('-y', dest='shifty', type=int, default = 200, help='shift y pixels for image after prewarping')
    args = parse.parse_args()

    if not args.input or not args.target:
        print("Invalid parameters")
        exit(-1)

    if not isinstance(args.sequence, int) or args.sequence <= 0:
        print("Sequence should be an positive integr")
        exit(-1)

    input, target = cv2.imread(args.input), cv2.imread(args.target)

    # extract feature points manually
    if not args.auto:
        input_points, target_points = read_feature_points(args.input), read_feature_points(args.target)
    else:
        # extract feature points automatically
        input_points_auto, target_points_auto = feature_points_detection(input, show=False), feature_points_detection(target, show=False)
        # eigth_points = [37, 40, 43, 46, 31, 49, 55, 58] # for test1 BAD DON'T USE 
        # eigth_points = [22, 23, 40, 43, 61, 65, 32, 38] # for test2 BAD DON'T USE 
        eigth_points = [18, 27, 40, 43, 49, 55, 6, 12] # for test3
        four_points = [37, 46, 49, 55]
        input_points_eight = [input_points_auto[p] for p in eigth_points]
        target_points_eight = [target_points_auto[p] for p in eigth_points]
        input_points_four = [input_points_auto[p] for p in four_points]
        target_points_four = [target_points_auto[p] for p in four_points]
        input_points, target_points = input_points_eight, target_points_eight

    # prewarping
    input_points_normailzed, input_t_norm = normalize(input_points)
    target_points_normailzed, target_t_norm = normalize(target_points)

    F = compute_fundamental_matrix(input_points_normailzed, target_points_normailzed, input_t_norm, target_t_norm)

    H0, H1 = compute_prewarp(F)
    
    input_pre_size = int(np.sqrt(input.shape[0]**2 + input.shape[1]**2) * 1.25) # for test1, 3
    target_pre_size = int(np.sqrt(target.shape[0]**2 + target.shape[1]**2) * 1.25)
    # input_pre_size = int(np.sqrt(input.shape[0]**2 + input.shape[1]**2) * 1.5) # for test2 BAD DON'T USE 
    # target_pre_size = int(np.sqrt(target.shape[0]**2 + target.shape[1]**2) * 1.5)

    input_prewarping, input_corners = do_projective_warping(input, H0, input_pre_size, input_pre_size, corners=True, shiftx=args.shiftx, shifty=args.shifty)
    target_prewarping, target_corners = do_projective_warping(target, H1, target_pre_size, target_pre_size, corners=True, shiftx=args.shiftx, shifty=args.shifty)
    
    cv2.imwrite(args.input.split('/')[-1].split('.')[0] + '_prewarping.png', input_prewarping)
    cv2.imwrite(args.target.split('/')[-1].split('.')[0] + '_prewarping.png', target_prewarping)

    # DEBUD prewarping
    # cv2.imshow("prewarpingi1", input_prewarping)
    # cv2.imshow("prewarpingt1", target_prewarping)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # features detection
    input_prewarping_points = feature_points_detection(input_prewarping, show=False)
    target_prewarping_points = feature_points_detection(target_prewarping, show=False)

    # DEBUD features detection 
    # input_prewarping_points = feature_points_detection(cv2.imread(args.input.split('/')[-1].split('.')[0] + '_prewarping.png'), show=False)
    # target_prewarping_points = feature_points_detection(cv2.imread(args.target.split('/')[-1].split('.')[0] + '_prewarping.png'), show=False)

    # delaunay triangulation
    points_input, trai_input = delaunay(input_prewarping, input_prewarping_points, input_corners, show=False)
    points_target, trai_target = delaunay(target_prewarping, target_prewarping_points, target_corners, show=False)
    
    # DEBUD delaunay triangulation
    # points_input, trai_input = delaunay(cv2.imread(args.input.split('/')[-1].split('.')[0] + '_prewarping.png'), input_prewarping_points, show=True)
    # points_target, trai_target = delaunay(cv2.imread(args.target.split('/')[-1].split('.')[0] + '_prewarping.png'), target_prewarping_points, show=True)

    # plot input image
    plt.figure(figsize=(20, 12))
    plt.subplot(1, args.sequence + 2, 1)
    plt.imshow(input[:, :, ::-1])
    plt.title('Origin Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    # view morphing and postwarping
    
    for i in range(args.sequence):
        alpha = round((i + 1) / (args.sequence + 1), 2)
        res = morphing(input_prewarping, points_input, trai_input, target_prewarping, points_target, trai_target, alpha)

        # control points detection
        # control_points = feature_points_detection(res, show=False)
        # control_points_four = [control_points[p] for p in four_points]
        # postwarping_four = [(int((1 - alpha) * input_points_four[p][0] + alpha * target_points_four[p][0]), 
        #                        int((1 - alpha) * input_points_four[p][1] + alpha * target_points_four[p][1])) for p in range(4)]
        # co = solve_prjective(control_points_four, postwarping_four)
        # res, _ = do_projective_warping(res, co, res.shape[0], res.shape[1], corners=False, shiftx=0, shifty=0)

        control_points_four = [(int((1 - alpha) * input_corners[p][0] + alpha * target_corners[p][0]), 
                             int((1 - alpha) * input_corners[p][1] + alpha * target_corners[p][1])) for p in range(4)]
        new_height = max(input.shape[0], target.shape[0])
        new_width = max(input.shape[1], target.shape[1])
        postwarping_four = [(0, 0), (0, new_height - 1), (new_width - 1, 0), (new_width - 1, new_height - 1)]
        co = solve_prjective(control_points_four, postwarping_four)
        res, _ = do_projective_warping(res, co, new_height, new_width, corners=False, shiftx=0, shifty=0)

        plt.subplot(1, args.sequence + 2, i + 2)
        plt.imshow(res[:, :, ::-1])
        plt.title('Morphing \\alpha={:.2f}'.format(alpha), fontsize=12)
        plt.xticks([])
        plt.yticks([])
    
    # plot target image
    plt.subplot(1, args.sequence + 2, args.sequence + 2)
    plt.imshow(target[:, :, ::-1])
    plt.title('Target Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()
