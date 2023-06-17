import numpy as np
from cv2 import cv2
# from scipy.linalg import solve 
from scipy import optimize


source, target = 'media/source.jpg', 'media/target.jpg'


def projective(x):
    return np.array([
        192-(x[2]*1)/(1*1),
        195-(x[5]*1)/(1*1),
        170-(x[1]*524+x[2]*1)/(x[7]*524 + 1*1),
        315-(x[4]*524+x[5]*1)/(x[7]*524 + 1*1),
        536-(x[0]*699+x[2]*1)/(x[6]*699+1*1),
        264-(x[3]*699+x[5]*1)/(x[6]*699+1*1),
        509-(x[0]*699+x[1]*524+x[2]*1)/(x[6]*699+x[7]*524+1*1),
        389-(x[3]*699+x[4]*524+x[5]*1)/(x[6]*699+x[7]*524+1*1)
    ])


def main():
    source_img = cv2.imread(source)
    target_img = cv2.imread(target)

    # source_coordinate = np.array([
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1],
    #     [0, 524, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 524, 1],
    #     [699, 524, 1, 0, 0, 0],
    #     [0, 0, 0, 699, 524, 1]
    # ])
    # target_coordinate = np.array([192, 195, 170, 315, 509, 389])
    # co = solve(source_coordinate, target_coordinate)
    # co = np.append(co, [0, 0, 1]).reshape(3, 3)

    co = optimize.fsolve(projective, np.zeros(8))
    co = np.append(co, 1).reshape(3, 3)
    source_heigth = np.shape(source_img)[0]
    source_width = np.shape(source_img)[1]
    
    for i in range(source_heigth):
        for j in range(source_width):
            new_pos = np.dot(co, np.array([j, i, 1]))
            new_pos /= new_pos[2]
            new_pos = new_pos.astype(np.int)
            target_img[new_pos[1], new_pos[0], :] = source_img[i, j, :]
            
    cv2.imwrite('results/projective.jpg', target_img)
    cv2.imshow('Projectve Warping', target_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
