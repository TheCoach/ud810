import cv2 as cv
import numpy as np
import os
import math

def local_maximum(m, NHoodSize):
    points = []
    while True:
        ind = np.unravel_index(np.argmax(m), m.shape)
        if (m[ind] == 0):
            break
        points.append(ind) 
        nHoodRbegin = max(0, ind[0] - math.floor(NHoodSize / 2))
        nHoodRend = min(m.shape[0] - 1, ind[0] + math.floor(NHoodSize / 2))
        nHoodCbegin = max(0, ind[1] - math.floor(NHoodSize / 2))
        nHoodCend = min(m.shape[1] - 1, ind[1] + math.floor(NHoodSize / 2))
        m[nHoodRbegin : nHoodRend + 1, nHoodCbegin : nHoodCend + 1] = 0
    return np.array(points)

def drawlines(img1, lines, pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    for r, pt1 in zip(lines, pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]]) # from line-point duality
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]]) # from line-point duality
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1, tuple(pt1), 5, color,-1)
    return img1

imgTransA = cv.imread(os.path.join('input', 'transA.jpg'))
imgTransB = cv.imread(os.path.join('input', 'transB.jpg'))
imgSimA = cv.imread(os.path.join('input', 'simA.jpg'))
imgSimB = cv.imread(os.path.join('input', 'simB.jpg'))

grayTransA = cv.cvtColor(imgTransA, cv.COLOR_BGR2GRAY)
grayTransB = cv.cvtColor(imgTransB, cv.COLOR_BGR2GRAY)
graySimA = cv.cvtColor(imgSimA, cv.COLOR_BGR2GRAY)
graySimB = cv.cvtColor(imgSimB, cv.COLOR_BGR2GRAY)

# dstTransA = cv.cornerHarris(grayTransA, 5, 9, 0.04)
# dstTransB = cv.cornerHarris(grayTransB, 5, 9, 0.04)
# _, dstTransA = cv.threshold(dstTransA, 0.2 * dstTransA.max(), 255, cv.THRESH_BINARY)
# _, dstTransB = cv.threshold(dstTransB, 0.2 * dstTransB.max(), 255, cv.THRESH_BINARY)
# dstTransA = np.uint8(dstTransA)
# dstTransB = np.uint8(dstTransB)
# cornersTransA = local_maximum(dstTransA, 15)

sift = cv.xfeatures2d.SIFT_create(nfeatures=30)
kpTransA, desTransA = sift.detectAndCompute(grayTransA, None)
kpTransB, desTransB = sift.detectAndCompute(grayTransB, None)
kpSimA, desSimA = sift.detectAndCompute(graySimA, None)
kpSimB, desSimB = sift.detectAndCompute(graySimB, None)

imgSiftTransA = cv.drawKeypoints(imgTransA, kpTransA, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgSiftTransB = cv.drawKeypoints(imgTransB, kpTransB, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgSiftSimA = cv.drawKeypoints(imgSimA, kpSimA, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgSiftSimB = cv.drawKeypoints(imgSimB, kpSimB, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img2a1 = np.concatenate((imgSiftTransA, imgSiftTransB), axis=1);
img2a2 = np.concatenate((imgSiftSimA, imgSiftSimB), axis=1);

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
matchesTrans = matcher.match(desTransA, desTransB)
matchesSim = matcher.match(desSimA, desSimB)
img2b1 = np.zeros_like(img2a1.shape)
img2b2 = np.zeros_like(img2a2.shape)

img2b1 = cv.drawMatches(imgTransA, kpTransA, imgTransB, kpTransB, matchesTrans, None)
img2b2 = cv.drawMatches(imgSimA, kpSimA, imgSimB, kpSimB, matchesSim, None)

pointsSimA = np.zeros((len(matchesSim), 2), dtype=np.float32)
pointsSimB = np.zeros((len(matchesSim), 2), dtype=np.float32)

for i, match in enumerate(matchesSim):
    pointsSimA[i, :] = kpSimA[match.queryIdx].pt
    pointsSimB[i, :] = kpSimB[match.trainIdx].pt

# Use fundamental matrix to warp B -> A
h, _ = cv.findHomography(pointsSimB, pointsSimA, cv.RANSAC);
graySimBWarped = cv.warpPerspective(graySimB, h, (imgSimA.shape[1], imgSimA.shape[0]));

imgSimOverlay = np.zeros_like(imgSimA); # B, G, R
imgSimOverlay[:,:,2] = graySimA;
imgSimOverlay[:,:,1] = graySimBWarped;

cv.imwrite(os.path.join('output', 'ps4-2-a-1.png'), img2a1)
cv.imwrite(os.path.join('output', 'ps4-2-a-2.png'), img2a2)
cv.imwrite(os.path.join('output', 'ps4-2-b-1.png'), img2b1)
cv.imwrite(os.path.join('output', 'ps4-2-b-2.png'), img2b2)
cv.imwrite(os.path.join('output', 'ps4-2-d-1.png'), graySimBWarped)
cv.imwrite(os.path.join('output', 'ps4-2-d-2.png'), imgSimOverlay)

# Extra, check epi lines in transA and transB
pointsTransA = np.zeros((len(matchesTrans), 2), dtype=np.float32)
pointsTransB = np.zeros((len(matchesTrans), 2), dtype=np.float32)
for i, match in enumerate(matchesTrans):
    pointsTransA[i, :] = kpTransA[match.queryIdx].pt
    pointsTransB[i, :] = kpTransB[match.trainIdx].pt

f, _ = cv.findFundamentalMat(pointsTransA, pointsTransB, cv.FM_RANSAC);
linesTransA = cv.computeCorrespondEpilines(pointsTransB, 2, f)
linesTransA = linesTransA.reshape(-1,3)
imgTransAEpi = drawlines(grayTransA, linesTransA, pointsTransA)

linesTransB = cv.computeCorrespondEpilines(pointsTransA, 1, f)
linesTransB = linesTransB.reshape(-1,3)
imgTransBEpi = drawlines(grayTransB, linesTransB, pointsTransB)

img = np.concatenate((imgTransAEpi, imgTransBEpi), axis=1);
cv.imwrite(os.path.join('output', 'ps4-epi.png'), img)

