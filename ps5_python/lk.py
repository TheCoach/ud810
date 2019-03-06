import cv2 as cv
import numpy as np
import math
import logging

def basic_lk(gray_img_1, gray_img_2, window_size = 5, t = 0.1):
    if (gray_img_1.ndim != 2 or gray_img_2.ndim != 2):
        raise Exception('gray_img_1 and gray_img_2 must be grayscale')

    if (gray_img_1.shape != gray_img_2.shape):
        raise Exception('gray_img_1 and gray_img_2 must be of the same shape')

    if (window_size % 2 != 1):
        raise Exception('window_size must be odd number')

    half_w = int((window_size - 1) / 2)

    gray_img_1 = np.pad(gray_img_1, half_w, 'constant')
    gray_img_2 = np.pad(gray_img_2, half_w, 'constant')

    ix = cv.Sobel(gray_img_1, cv.CV_64F, 1, 0)
    iy = cv.Sobel(gray_img_1, cv.CV_64F, 0, 1)
    it = np.float64(gray_img_2) - np.float64(gray_img_1)

    ixx = ix * ix
    ixy = ix * iy
    iyy = iy * iy
    ixt = ix * it
    iyt = iy * it
    ixx = cv.GaussianBlur(ixx, (window_size, window_size), 0)
    ixy = cv.GaussianBlur(ixy, (window_size, window_size), 0)
    iyy = cv.GaussianBlur(iyy, (window_size, window_size), 0)
    ixt = cv.GaussianBlur(ixt, (window_size, window_size), 0)
    iyt = cv.GaussianBlur(iyt, (window_size, window_size), 0)
    
    u = np.zeros_like(gray_img_1, dtype = np.float64)
    v = np.zeros_like(gray_img_1, dtype = np.float64)

    for r in range(half_w, gray_img_1.shape[0] - half_w):
        for c in range(half_w, gray_img_2.shape[1] - half_w):
            M = np.array([[ixx[r, c], ixy[r, c]], [ixy[r, c], iyy[r, c]]], dtype = np.float64)
            w, _ = np.linalg.eig(M)
            w = np.sort(w)[::-1]
            if (np.linalg.matrix_rank(M) != 2 or w[0] / w[1] < t):
                continue

            b = np.array([[-ixt[r, c]], [-iyt[r, c]]], dtype = np.float64)
            d = np.matmul(np.linalg.inv(M), b)
            assert(d.shape == (2, 1))
            u[r, c] = d[0, 0]
            v[r, c] = d[1, 0]

    u = u[half_w : u.shape[0] - half_w, half_w : u.shape[1] - half_w]
    v = v[half_w : v.shape[0] - half_w, half_w : v.shape[1] - half_w]

    return u, v

def hi_lk(img_1, img_2, n, window_size = 5):
    if (math.floor(n) != n):
        raise Exception('n must be an integer')

    if (img_1.ndim != 2 or img_2.ndim != 2):
        raise Exception('img_1 and img_2 must be grayscale')

    if (img_1.shape != img_2.shape):
        raise Exception('img_1 and img_2 must be of the same shape')

    max_levels = math.floor(math.log(min(img_1.shape[0], img_1.shape[1])/2, 2))
    if (n > max_levels):
        print(f'hi_lk(), {__file__}: n ({n}) is too big, overwrite to {max_levels}')

    n = max_levels

    g_img_1 = []
    g_img_1.append(img_1)
    g_img_2 = []
    g_img_2.append(img_2)
    for i in range(1, n + 1):
        g_img_1.append(cv.pyrDown(g_img_1[i - 1]))
        g_img_2.append(cv.pyrDown(g_img_2[i - 1]))

    u = np.zeros_like(g_img_1[n], dtype = np.float64)
    v = np.zeros_like(g_img_1[n], dtype = np.float64)

    for k in range(n, -1, -1):
        if k != n:
            h, w = map(int, g_img_1[k].shape)
            u = 2 * cv.pyrUp(u, dstsize = (w, h))
            v = 2 * cv.pyrUp(v, dstsize = (w, h))
        wk = warp(g_img_1[k], u, v)
        dx, dy = basic_lk(wk, g_img_2[k], window_size)
        u = u + dx
        v = v + dy

    return u, v;

'''
The  two visulisation methods are borrowed from OpenCV tutorial
'''
def draw_flow_grid(img, u, v, step = 16):
    # scale displacements to be visible
    u = 5 * u
    v = 5 * v

    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2,-1).astype(int)
    fx = u[y,x].T
    fy = v[y,x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_flow_hsv(u, v):
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype = np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

def warp(img, u, v):
    h, w = map(int, img.shape)
    y, x = np.mgrid[0 : h : 1, 0 : w : 1].astype(np.float32)
    warped = cv.remap(img, x + u.astype(np.float32), y + v.astype(np.float32), interpolation = cv.INTER_NEAREST)
    warpedL = cv.remap(img, x + u.astype(np.float32), y + v.astype(np.float32), interpolation = cv.INTER_LINEAR)
    I = warped == 0
    warped[I] = warpedL[I]
    return warped