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
    

def twoDCTBlock(img, sample=None, quan=None):
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
        for i in range(remain, img.shape[0]):
            for j in range(remain, img.shape[1]):
                img_dct[i, j] = 0

    if isinstance(quan, np.ndarray):
        img_dct_quan = quantization(img_dct, quan)
        img_dct = dequantization(img_dct_quan, quan)
        
    img_idct = np.dot(np.dot(co.T, img_dct), co)
    return img_dct, img_idct


def twoDCT(img, block=8, sample=None, quan=None):
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
            bimg_dct, bimg_idct = twoDCTBlock(bimg, sample, quan)
            wimg_dct.append(bimg_dct)
            wimg_idct.append(bimg_idct)
        himg_dct.append(np.hstack(wimg_dct))
        himg_idct.append(np.hstack(wimg_idct))

    img_dct = np.vstack(himg_dct)[:img.shape[0], :img.shape[1]]
    img_idct = np.vstack(himg_idct)[:img.shape[0], :img.shape[1]]
    psnr = PSNR(img, img_idct)
    return img_dct, img_idct, psnr


def quantization(dct, quan):
    qdct = np.zeros_like(dct)
    for i in range(qdct.shape[0]):
        for j in range(qdct.shape[1]):
            qdct[i, j] = int(np.round(dct[i, j] / quan[i, j]))
    return qdct


def dequantization(qdct, quan):
    dct = np.zeros_like(qdct)
    for i in range(dct.shape[0]):
        for j in range(dct.shape[1]):
            dct[i, j] = int(qdct[i, j] * quan[i, j])
    return dct


def cmpWithOrNotQuan(img, quan):

    # 2 dct without quan
    img_dct, img_idct, psnr = twoDCT(img, block=8)

    plt.figure(figsize=(20, 12))
    # plot source image
    plt.subplot(2, 3, 1)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 2 dct
    plt.subplot(2, 3, 2)
    plt.title("2DCT BLOCK=8*8 Quan=False")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(2, 3, 3)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])


    # 2 dct with quan
    img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)

    # plot source image
    plt.subplot(2, 3, 4)
    plt.title("Source Image")
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
     
    # plot 2 dct
    plt.subplot(2, 3, 5)
    plt.title("2DCT BLOCK=8*8 Quan=True")
    plt.imshow(np.log(abs(img_dct)))
    plt.xticks([])
    plt.yticks([])

    # plot 2 idct
    plt.subplot(2, 3, 6)
    plt.title("IDCT PSNR={:.2f}".format(psnr))
    plt.imshow(img_idct, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.show()


def cmpAandQ(img, quans):
    psnr_Q = []
    psnr_Canon = []
    psnr_Nikon = []
    for i in np.arange(0.1, 2.1, 0.1):
        # Q
        quan = i * quans[0]
        img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)
        psnr_Q.append(psnr)
        # Canon
        quan = i * quans[1]
        img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)
        psnr_Canon.append(psnr)
        # Nikon
        quan = i * quans[2]
        img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)
        psnr_Nikon.append(psnr)
        
    plt.plot(np.arange(0.1, 2.1, 0.1), psnr_Q, color='royalblue', label='Q')
    plt.plot(np.arange(0.1, 2.1, 0.1), psnr_Canon, color='cyan', label='Canon')
    plt.plot(np.arange(0.1, 2.1, 0.1), psnr_Nikon, color='orange', label='Nikon')
    plt.xlabel("a")
    plt.ylabel("PSNR")
    plt.legend()
    plt.show()
        

def cmpAndQShow(img, quans):

    plt.figure(figsize=(20, 12))
    for num, i in enumerate([0.5, 1.0, 1.5]):
        # Q
        quan = i * quans[0]
        img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)
        plt.subplot(3, 6, num * 2 + 1)
        plt.title("DCT BLOCK=8*8 {:.1f}*Q".format(i))
        plt.imshow(np.log(abs(img_dct)))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 6, num * 2 + 2)
        plt.title("IDCT PSNR={:.2f}".format(psnr))
        plt.imshow(img_idct, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Canon
        quan = i * quans[1]
        img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)
        plt.subplot(3, 6, num * 2 + 7)
        plt.title("DCT BLOCK=8*8 {:.1f}*Canon".format(i))
        plt.imshow(np.log(abs(img_dct)))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 6, num * 2 + 8)
        plt.title("IDCT PSNR={:.2f}".format(psnr))
        plt.imshow(img_idct, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # Nikon
        quan = i * quans[2]
        img_dct, img_idct, psnr = twoDCT(img, block=8, quan=quan)
        plt.subplot(3, 6, num * 2 + 13)
        plt.title("DCT BLOCK=8*8 {:.1f}*Nikon".format(i))
        plt.imshow(np.log(abs(img_dct)))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 6, num * 2 + 14)
        plt.title("IDCT PSNR={:.2f}".format(psnr))
        plt.imshow(img_idct, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        
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
    img_gray = bgr2gray(img)

    Q = np.asarray([
         [16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    Canon = np.asarray([
         [1, 1, 1, 2, 3, 6, 8, 10],
         [1, 1, 2, 3, 4, 8, 9, 8],
         [2, 2, 2, 3, 6, 8, 10, 8],
         [2, 2, 3, 4, 7, 12, 11, 9],
         [3, 3, 8, 11, 10, 16, 15, 11],
         [3, 5, 8, 10, 12, 15, 16, 13],
         [7, 10, 11, 12, 15, 17, 17, 14],
         [14, 13, 13, 15, 15, 14, 14, 14]
    ])

    Nikon = np.asarray([
         [2, 1, 1, 2, 3, 5, 6, 7],
         [1, 1, 2, 2, 3, 7, 7, 7],
         [2, 2, 2, 3, 5, 7, 8, 7],
         [2, 2, 3, 3, 6, 10, 10, 7],
         [2, 3, 4, 7, 8, 13, 12, 9],
         [3, 4, 7, 8, 10, 12, 14, 11],
         [6, 8, 9, 10, 12, 15, 14, 12],
         [9, 11, 11, 12, 13, 12, 12, 12]
    ])

    # task0
    cmpWithOrNotQuan(img_gray, Q)

    # task1
    quans = [Q, Canon, Nikon]
    cmpAandQ(img_gray, quans=quans)
    cmpAndQShow(img_gray, quans=quans)


if __name__ == '__main__':
    main()
