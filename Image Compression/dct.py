import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import argparse


def bgr2gray(img):
    img_gray = np.zeros((img.shape[0], img.shape[1])).astype(np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gray[i, j] = (15 * img[i, j, 0] + 75 * img[i, j, 1] + 38 * img[i, j, 2]) >> 7
    return img_gray


def PSNR(img, img_idct):
    MSE = np.sum((img_idct - img)**2)/(img.shape[0]*img.shape[1])
    PSNR = 10 * np.log(255**2/MSE)
    return PSNR


def test(img):
    # plot source image
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    # 2 dct, idct, psnr
    img_dct, img_idct, psnr = twoDCT(img)
     
    # plot 2 dct
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(1, 3, 3)
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
    plt.show()


def oneDCT(img, sample=None):
    co = np.zeros_like(img)
    co[0, :] = np.sqrt(1/img.shape[0])

    for i in range(1, img.shape[0]):
        for j in range(img.shape[1]):
            co[i, j] = np.cos((2 * j + 1) * np.pi * i / (2 * img.shape[0])) * np.sqrt(2 / img.shape[0])
    
    # row
    row = []
    img_row = np.hsplit(img, img.shape[1])
    for i in img_row:
        dct_row = np.dot(co, i)
        row.append(dct_row)
    dct_after_row = np.hstack(row)
    if sample != None:
        for i in range(sample, dct_after_row.shape[0]):
            for j in range(dct_after_row.shape[1]):
                dct_after_row[i, j] = 0
    #col
    col = []
    img_col = np.vsplit(dct_after_row, img.shape[0])
    for j in img_col:
        dct_col = np.dot(j, co.T)
        col.append(dct_col)
    img_dct = np.vstack(col)
    # img_dct_copy = img_dct.copy()
    if sample != None:
        for i in range(img_dct.shape[0]):
            for j in range(sample, img_dct.shape[1]):
                img_dct[i, j] = 0
    # idct
    img_idct = np.dot(np.dot(co.T, img_dct), co)
    # img_idct_1_2 = np.dot(np.dot(co.T, img_dct_copy), co)

    # if sample != None:
    #     plt.figure(figsize=(20, 12))
    #     plt.subplot(1, 5, 1)
    #     plt.title("Source Image")
    #     plt.imshow(img, cmap="gray")
    #     plt.xticks([])
    #     plt.yticks([])

    #     plt.subplot(1, 5, 2)
    #     plt.title("DCT 1/2")
    #     plt.imshow(np.log(abs(dct_after_row)))
    #     plt.xticks([])
    #     plt.yticks([])

    #     plt.subplot(1, 5, 3)
    #     plt.title("IDCT 1/2 PSNR={:.2f}".format(PSNR(img, img_idct_1_2)))
    #     plt.imshow(img_idct_1_2, cmap="gray")
    #     plt.xticks([])
    #     plt.yticks([])

    #     plt.subplot(1, 5, 4)
    #     plt.title("DCT 1/4")
    #     plt.imshow(np.log(abs(img_dct)))
    #     plt.xticks([])
    #     plt.yticks([])

    #     plt.subplot(1, 5, 5)
    #     plt.title("IDCT 1/4 PSNR={:.2f}".format(PSNR(img, img_idct)))
    #     plt.imshow(img_idct, cmap="gray")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()

    return img_dct, img_idct

    
def twoDCTBlock(img, sample=None):
    # cv2.dct, cv2.idct
    # img_dct = cv2.dct(img)  
    # img_idct = cv2.idct(img_dct)

    # my dct, idct
    co = np.zeros_like(img)
    co[0, :] = np.sqrt(1/img.shape[0])

    for i in range(1, img.shape[0]):
        for j in range(img.shape[1]):
            co[i, j] = np.cos((2 * j + 1) * np.pi * i / (2 * img.shape[0])) * np.sqrt(2 / img.shape[0])
    
    img_dct = np.dot(np.dot(co, img), co.T)

    if sample != None:
        remain = int(np.sqrt(sample))
        remain = img.shape[0] // remain
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i >= remain or j >= remain:
                    img_dct[i, j] = 0
        
    img_idct = np.dot(np.dot(co.T, img_dct), co)
    return img_dct, img_idct


def twoDCT(img, block=8, sample=None):
    # img_dct = np.zeros_like(img)
    # img_idct = np.zeros_like(img)
    
    h, w = img.shape
    if h % block != 0:
        h = int(np.ceil(h / block) * block)
    if w % block != 0:
        w = int(np.ceil(w / block) * block)
    img_padding = np.zeros((h, w))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_padding[i, j] = img[i, j]

    himg = np.vsplit(img_padding, h // block)
    himg_dct, himg_idct = [], []
    for i in range(h // block):
        wimg = np.hsplit(himg[i], w // block)
        wimg_dct, wimg_idct = [], []
        for j in range(w // block):
            bimg = wimg[j]
            bimg_dct, bimg_idct = twoDCTBlock(bimg, sample)
            wimg_dct.append(bimg_dct)
            wimg_idct.append(bimg_idct)
        himg_dct.append(np.hstack(wimg_dct))
        himg_idct.append(np.hstack(wimg_idct))

    img_dct = np.vstack(himg_dct)[:img.shape[0], :img.shape[1]]
    img_idct = np.vstack(himg_idct)[:img.shape[0], :img.shape[1]]
    psnr = PSNR(img, img_idct)
    return img_dct, img_idct, psnr


def cmpOneAndTwoDCT(img):

    # 1DCT without sample
    img_dct, img_idct = oneDCT(img)
    psnr = PSNR(img, img_idct)

    plt.figure(figsize=(20, 12))
    # plot source image
    plt.subplot(3, 3, 1)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 1 dct
    plt.subplot(3, 3, 2)
    plt.title("1DCT")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 1 idct
    plt.subplot(3, 3, 3)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
    # 2 dct
    img_dct, img_idct = twoDCTBlock(img)
    psnr = PSNR(img, img_idct)
    # plot source image
    plt.subplot(3, 3, 4)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 2 dct
    plt.subplot(3, 3, 5)
    plt.title("2DCT")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(3, 3, 6)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    # 2 dct 8*8
    img_dct, img_idct, psnr = twoDCT(img, block=8)
    # plot source image
    plt.subplot(3, 3, 7)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 2 dct
    plt.subplot(3, 3, 8)
    plt.title("2DCT BLOCK=8*8")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(3, 3, 9)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.show()


def cmpOneAndTwoDCTWithSample(img):

    # 1DCT without sample
    img_dct, img_idct = oneDCT(img)
    psnr = PSNR(img, img_idct)

    plt.figure(figsize=(20, 12))
    # plot source image
    plt.subplot(2, 6, 1)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 1 dct
    plt.subplot(2, 6, 2)
    plt.title("1DCT")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 1 idct
    plt.subplot(2, 6, 3)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
    # 2 dct with sample
    img_dct, img_idct = twoDCTBlock(img)
    psnr = PSNR(img, img_idct)
    # plot source image
    plt.subplot(2, 6, 4)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 2 dct
    plt.subplot(2, 6, 5)
    plt.title("2DCT")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(2, 6, 6)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    # 1DCT sample
    img_dct, img_idct = oneDCT(img, sample=int(img.shape[0]/2))
    psnr = PSNR(img, img_idct)

    # plot source image
    plt.subplot(2, 6, 7)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 1 dct
    plt.subplot(2, 6, 8)
    plt.title("1DCT(1/4)")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 1 idct
    plt.subplot(2, 6, 9)
    plt.title("IDCT PSNR={:.6f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
    # 2 dct with sample
    img_dct, img_idct = twoDCTBlock(img, sample=4)
    psnr = PSNR(img, img_idct)
    # plot source image
    plt.subplot(2, 6, 10)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 2 dct
    plt.subplot(2, 6, 11)
    plt.title("2DCT(1/4)")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(2, 6, 12)
    plt.title("IDCT PSNR={:.6f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.show()


def cmpCoefficients(img, sample=[4, 16, 64]):
    plt.figure(figsize=(20, 12))
    length = len(sample)
    for i, item in enumerate(sample):
        img_dct, img_idct, psnr = twoDCT(img, sample=item)
        # plot source image
        plt.subplot(length, 3, 3 * i + 1)
        plt.title("Source Image")
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # plot dct
        plt.subplot(length, 3, 3 * i + 2)
        plt.title("DCT(1/{})".format(item))
        plt.imshow(np.log(abs(img_dct)))
        plt.xticks([])
        plt.yticks([])

        # plot idct
        plt.subplot(length, 3, 3 * i + 3)
        plt.title("IDCT PSNR={:.2f}".format(psnr))
        plt.imshow(img_idct, cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.show()
    

def cmpBlockSize(img, maxN=8):
    psnr_list = []
    pos = 0
    plt.figure(figsize=(20, 12))
    for i in range(maxN):
        img_dct, img_idct, psnr = twoDCT(img, block=i + 1)
        # plot source image
        pos += 1
        plt.subplot(maxN // 2, 6, pos)
        plt.title("Source Image")
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # plot dct
        pos += 1
        plt.subplot(maxN // 2, 6, pos)
        plt.title("DCT BLOCK={}*{}".format(i + 1, i + 1))
        plt.imshow(np.log(abs(img_dct)))
        plt.xticks([])
        plt.yticks([])

        # plot idct
        pos += 1
        plt.subplot(maxN // 2, 6, pos)
        plt.title("IDCT PSNR={:.2f}".format(psnr))
        plt.imshow(img_idct, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        psnr_list.append(psnr)
    plt.show()

    plt.figure()
    plt.plot(np.array(list(range(maxN)))**2, psnr_list)
    plt.xlabel("Block Size")
    plt.ylabel("PSNR")
    plt.show()


def main():
    parse = argparse.ArgumentParser(description='Image Compression')
    parse.add_argument('-i', dest='input', type=str, default='lena.png', help='input picture')
    args = parse.parse_args()

    if not args.input:
        print("Invalid input picture")
        exit(-1)

    # to gray
    img = cv2.imread(args.input).astype(np.int64)
    # img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_CUBIC)
    img_gray = bgr2gray(img)

    # test
    # test(img_gray)

    # task1
    cmpOneAndTwoDCT(img_gray)
    cmpOneAndTwoDCTWithSample(img_gray)

    # task2
    # cmpCoefficients(img_gray, sample=[4, 16, 64])

    # task3
    # cmpBlockSize(img_gray, maxN=8)
    


if __name__ == '__main__':
    main()
