# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:23:26 2019

@author: yutian.yuan
"""

import lk

import sys
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

def p1():
    gray0 = cv.imread(os.path.join('input', 'TestSeq', 'Shift0.png'), cv.IMREAD_GRAYSCALE)
    gray1 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR2.png'), cv.IMREAD_GRAYSCALE)
    gray2 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR5U5.png'), cv.IMREAD_GRAYSCALE)
    gray3 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR10.png'), cv.IMREAD_GRAYSCALE)
    gray4 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR20.png'), cv.IMREAD_GRAYSCALE)
    gray5 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR40.png'), cv.IMREAD_GRAYSCALE)

    gray0 = cv.GaussianBlur(gray0, (11, 11), 0)
    gray1 = cv.GaussianBlur(gray1, (11, 11), 0)
    gray2 = cv.GaussianBlur(gray2, (11, 11), 0)
    gray3 = cv.GaussianBlur(gray3, (11, 11), 0)
    gray4 = cv.GaussianBlur(gray4, (11, 11), 0)

    u, v = lk.basic_lk(gray0, gray1, 11)
    vis = lk.draw_flow_grid(gray1, u, v)
    cv.imwrite(os.path.join('output', 'ps5-1-a-1.png'), vis)

    u, v = lk.basic_lk(gray0, gray2, 11)
    vis = lk.draw_flow_grid(gray2, u, v)
    cv.imwrite(os.path.join('output', 'ps5-1-a-2.png'), vis)

    u, v = lk.basic_lk(gray0, gray3, 11)
    vis = lk.draw_flow_grid(gray3, u, v)
    cv.imwrite(os.path.join('output', 'ps5-1-b-1.png'), vis)

    u, v = lk.basic_lk(gray0, gray4, 11)
    vis = lk.draw_flow_grid(gray4, u, v)
    cv.imwrite(os.path.join('output', 'ps5-1-b-2.png'), vis)

    u, v = lk.basic_lk(gray0, gray5, 11)
    vis = lk.draw_flow_grid(gray5, u, v)
    cv.imwrite(os.path.join('output', 'ps5-1-b-3.png'), vis)

def p2():
    g = []
    g.append(cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_01.jpg'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 4):
        g.append(cv.pyrDown(g[i - 1]))

    plt.figure()
    for i in range(0, 4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(g[i], cmap='gray')
        plt.title('G' + str(i))

    plt.savefig(os.path.join('output', 'ps5-2-a-1.png'))

    l = []
    l.append(g[3])
    for i in range(3, 0, -1):
        h, w = map(int, g[i - 1].shape)
        l.append(cv.subtract(g[i - 1], cv.pyrUp(g[i], dstsize=(w, h))))

    l.reverse()
    
    plt.figure()
    for i in range(0, 4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(l[i], cmap='gray')
    plt.savefig(os.path.join('output', 'ps5-2-b-1.png'))

def p3_a():
    g_img1 = []
    g_img1.append(cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_01.jpg'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 5):
        g_img1.append(cv.pyrDown(g_img1[i - 1]))

    g_img2 = []
    g_img2.append(cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_02.jpg'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 5):
        g_img2.append(cv.pyrDown(g_img2[i - 1]))

    g_img3 = []
    g_img3.append(cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_03.jpg'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 5):
        g_img3.append(cv.pyrDown(g_img3[i - 1]))

    G_LEVEL = 2

    u, v = lk.basic_lk(g_img1[G_LEVEL], g_img2[G_LEVEL])
    warpedI2 = lk.warp(g_img2[G_LEVEL], u, v)
    vis12 = lk.draw_flow_hsv(u, v)

    u, v = lk.basic_lk(g_img2[G_LEVEL], g_img3[G_LEVEL])
    vis23 = lk.draw_flow_hsv(u, v)
    vis = np.concatenate((vis12, vis23), axis=1);

    diffImg = cv.subtract(g_img1[G_LEVEL], warpedI2)

    cv.imwrite(os.path.join('output', 'ps5-3-a-1.png'), vis)
    cv.imwrite(os.path.join('output', 'ps5-3-a-2.png'), diffImg)

def p3_b():
    g_img1 = []
    g_img1.append(cv.imread(os.path.join('input', 'DataSeq2', '0.png'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 5):
        g_img1.append(cv.pyrDown(g_img1[i - 1]))

    g_img2 = []
    g_img2.append(cv.imread(os.path.join('input', 'DataSeq2', '1.png'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 5):
        g_img2.append(cv.pyrDown(g_img2[i - 1]))

    g_img3 = []
    g_img3.append(cv.imread(os.path.join('input', 'DataSeq2', '2.png'), cv.IMREAD_GRAYSCALE))
    for i in range(1, 5):
        g_img3.append(cv.pyrDown(g_img3[i - 1]))

    G_LEVEL = 3

    u, v = lk.basic_lk(g_img1[G_LEVEL], g_img2[G_LEVEL])
    warpedI2 = lk.warp(g_img2[G_LEVEL], u, v)
    vis12 = lk.draw_flow_hsv(u, v)

    u, v = lk.basic_lk(g_img2[G_LEVEL], g_img3[G_LEVEL])
    vis23 = lk.draw_flow_hsv(u, v)
    vis = np.concatenate((vis12, vis23), axis=1);

    diffImg = cv.subtract(g_img1[G_LEVEL], warpedI2)

    cv.imwrite(os.path.join('output', 'ps5-3-b-1.png'), vis)
    cv.imwrite(os.path.join('output', 'ps5-3-b-2.png'), diffImg)
    
def p4_a():
    i1 = cv.imread(os.path.join('input', 'TestSeq', 'Shift0.png'), cv.IMREAD_GRAYSCALE)
    i2 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR5U5.png'), cv.IMREAD_GRAYSCALE)
    i3 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR10.png'), cv.IMREAD_GRAYSCALE)
    i4 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR20.png'), cv.IMREAD_GRAYSCALE)
    i5 = cv.imread(os.path.join('input', 'TestSeq', 'ShiftR40.png'), cv.IMREAD_GRAYSCALE)

    u3,v3 = lk.hi_lk(i1, i3, 10)
    u4,v4 = lk.hi_lk(i1, i4, 10)
    u5,v5 = lk.hi_lk(i1, i5, 10)

    flow1 = lk.draw_flow_hsv(u3, v3)
    flow2 = lk.draw_flow_hsv(u4, v4)
    flow3 = lk.draw_flow_hsv(u5, v5)
    
    warped_i3 = lk.warp(i3, u3, v3)
    warped_i4 = lk.warp(i4, u4, v4)
    warped_i5 = lk.warp(i5, u5, v5)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(cv.cvtColor(flow1, cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(flow2, cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(flow3, cv.COLOR_BGR2RGB))

    plt.savefig(os.path.join('output', 'ps5-4-a-1.png'))

    plt.figure()
    plt.axis("off")
    plt.subplot(1, 3, 1)
    plt.imshow(cv.subtract(i1, warped_i3), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(cv.subtract(i1, warped_i4), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(cv.subtract(i1, warped_i5), cmap='gray')

    plt.savefig(os.path.join('output', 'ps5-4-a-2.png'))
    
    u2, v2 = lk.hi_lk(i1, i2, 10)
    flow4 = lk.draw_flow_grid(i2, u2, v2)
    cv.imwrite(os.path.join('output', 'ps5-4-a-3.png'), flow4)

def p4_b():
    i1 = cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_01.jpg'), cv.IMREAD_GRAYSCALE)
    i2 = cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_02.jpg'), cv.IMREAD_GRAYSCALE)
    i3 = cv.imread(os.path.join('input', 'DataSeq1', 'yos_img_03.jpg'), cv.IMREAD_GRAYSCALE)

    u2,v2 = lk.hi_lk(i1, i2, 10)
    u3,v3 = lk.hi_lk(i2, i3, 10)
    flow2 = lk.draw_flow_hsv(u2, v2)
    flow3 = lk.draw_flow_hsv(u3, v3)
    
    warped_i2 = lk.warp(i2, u2, v2)
    warped_i3 = lk.warp(i3, u3, v3)

    cv.imwrite(os.path.join('output', 'ps5-4-b-1.png'), flow2)
    cv.imwrite(os.path.join('output', 'ps5-4-b-2.png'), flow3)
    cv.imwrite(os.path.join('output', 'ps5-4-b-3.png'), cv.subtract(i1, warped_i2))
    cv.imwrite(os.path.join('output', 'ps5-4-b-4.png'), cv.subtract(i2, warped_i3))

def p4_c():
    i1 = cv.imread(os.path.join('input', 'DataSeq2', '0.png'), cv.IMREAD_GRAYSCALE)
    i2 = cv.imread(os.path.join('input', 'DataSeq2', '1.png'), cv.IMREAD_GRAYSCALE)
    i3 = cv.imread(os.path.join('input', 'DataSeq2', '2.png'), cv.IMREAD_GRAYSCALE)

    u2,v2 = lk.hi_lk(i1, i2, 10, window_size=11)
    flow2 = lk.draw_flow_hsv(u2, v2)
    u3,v3 = lk.hi_lk(i2, i3, 10, window_size=11)
    flow3 = lk.draw_flow_hsv(u3, v3)
    warped_i2 = lk.warp(i2, u2, v2)
    warped_i3 = lk.warp(i3, u3, v3)
    cv.imwrite(os.path.join('output', 'ps5-4-c-1.png'), flow2)
    cv.imwrite(os.path.join('output', 'ps5-4-c-2.png'), flow3)
    cv.imwrite(os.path.join('output', 'ps5-4-c-3.png'), cv.subtract(i1, warped_i2))
    cv.imwrite(os.path.join('output', 'ps5-4-c-4.png'), cv.subtract(i2, warped_i3))

def main(argv):
    p4_c()

if __name__ == '__main__':
    main(sys.argv[1:])
