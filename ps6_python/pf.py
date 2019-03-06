import numpy as np
import cv2 as cv
import math
import copy

def mse(temp, img, x, y):
    if (temp.ndim == 3):
        temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

    if (img.ndim == 3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if (math.floor(x) != x or math.floor(y) != y):
        print(f'mse(), {__file__}: x or y is not integer')

    x = int(x)
    y = int(y)

    if (x > img.shape[1] - 1 or x < 0 or y > img.shape[0] - 1 or y < 0):
        print(f'x=={x} or y=={y} is out of bound, img\'s shape is {img.shape}')
        return np.finfo(np.float64).max

    half_w = int(temp.shape[1] / 2)
    half_h = int(temp.shape[0] / 2)

    x = max(x, half_w)
    x = min(x, img.shape[1] - half_w - 1)

    y = max(y, half_h)
    y = min(y, img.shape[0] - half_h - 1)

    w_xbegin = x - half_w
    w_xend = x + half_w + 1
    w_ybegin = y - half_h
    w_yend = y + half_h + 1

    w = img[w_ybegin : w_yend, w_xbegin : w_xend]
    return np.square(np.subtract(temp, w, dtype=np.float64)).mean()

class ParticleFilter:
    def __init__(self, cx, cy, template, num_particles = 500, mse_sigma = 10, dynamic_sigma = 10, init_nhood_size = 25):
        if (template.ndim == 3):
            self._MODEL = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        else:
            self._MODEL = copy.deepcopy(template)
        self._NUM_PARTICLES = num_particles
        self._MSE_SIGMA = mse_sigma
        self._DYNAMIC_COV_SIGMA = dynamic_sigma

        DX, DY = np.mgrid[-int(init_nhood_size/2) : int(init_nhood_size/2) + 1, -int(init_nhood_size/2) : int(init_nhood_size/2) + 1].astype(int)
        px, py = np.unravel_index(np.random.choice(init_nhood_size * init_nhood_size, self._NUM_PARTICLES), (init_nhood_size, init_nhood_size))
        self._s = np.empty((3, self._NUM_PARTICLES), dtype = np.float64)
        # s = 3xN
        # s(0, _): x
        # s(1, _): y
        # s(2, _): weight
        for i, (x, y) in enumerate(zip(cx + DX[px, py], cy + DY[px, py])):
            self._s[0][i] = x
            self._s[1][i] = y
            self._s[2][i] = 1 / self._NUM_PARTICLES
        assert(abs(1 - self._s[2].sum()) < 1e-05)

    def filter(self, image, model_update_alpha = 0):
        # resample according to weights at (t-1)
        resample_index = np.random.choice(self._NUM_PARTICLES, self._NUM_PARTICLES, replace = True, p = self._s[2])

        # action (dynamics system) model, i.e. some guassian noise on each particles
        # p(x_t | x_t-1, u_t)
        ux, uy = np.random.multivariate_normal([0, 0], [[self._DYNAMIC_COV_SIGMA, 0], [0, self._DYNAMIC_COV_SIGMA]], len(self._s[2])).T
        self._s[0] = np.ceil(self._s[0][resample_index] + ux[resample_index])
        self._s[1] = np.ceil(self._s[1][resample_index] + uy[resample_index])

        for i, (x_t, y_t) in enumerate(zip(self._s[0], self._s[1])):
            # w_t = p(z_t|x_t)
            m = mse(self._MODEL, image, x_t, y_t)
            w_t = np.exp(-m / (2 * self._MSE_SIGMA ** 2))
            self._s[2][i] = w_t

        self._s[2] = self._s[2] / self._s[2].sum()
        assert(abs(1 - self._s[2].sum()) < 1e-05)

        cx, cy = int(round(np.average(self._s[0], weights=self._s[2]))), int(round(np.average(self._s[1], weights=self._s[2])))
        if (model_update_alpha != 0):
            self._update_model(cx, cy, image, model_update_alpha)
            cv.imshow('new model', self._MODEL)

        return cx, cy

    def visualize(self, frame, verbose = 2):
        CV_COLOR_GREEN = (0, 255, 0)
        CV_COLOR_YELLOW = (0, 255, 255)
        CV_COLOR_RED = (0, 0, 255)
        CV_COLOR_LIGHT_CYAN = (255, 255, 102)

        cx, cy = int(round(np.average(self._s[0], weights=self._s[2]))), int(round(np.average(self._s[1], weights=self._s[2])))
        half_model_w = math.floor(self._MODEL.shape[1] / 2)
        half_model_h = math.floor(self._MODEL.shape[0] / 2)
        cv.rectangle(frame, (cx -half_model_w, cy - half_model_h), (cx + half_model_w, cy + half_model_h), CV_COLOR_GREEN)
        cv.circle(frame, (cx, cy), 3, CV_COLOR_RED, -1)

        if (verbose > 1):
            d = 0.0
            for x, y, w in zip(self._s[0], self._s[1], self._s[2]):
                x = int(x)
                y = int(y)
                d += ((x - cx) ** 2 + (y - cy) ** 2) * w
                cv.circle(frame, (x, y), 1, CV_COLOR_YELLOW, -1)
            cv.circle(frame, (cx, cy), int(round(d)), CV_COLOR_LIGHT_CYAN, 1)


    def _update_model(self, x, y, image, model_update_alpha):
        half_w = int(self._MODEL.shape[1] / 2)
        half_h = int(self._MODEL.shape[0] / 2)

        x = max(x, half_w)
        x = min(x, image.shape[1] - half_w - 1)
        y = max(y, half_h)
        y = min(y, image.shape[0] - half_h - 1)

        w_xbegin = x - half_w
        w_xend = x + half_w + 1
        w_ybegin = y - half_h
        w_yend = y + half_h + 1

        self._MODEL = cv.addWeighted(image[w_ybegin : w_yend, w_xbegin : w_xend], model_update_alpha, self._MODEL, 1 - model_update_alpha, 0)