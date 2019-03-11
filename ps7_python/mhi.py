import cv2 as cv
import numpy as np
import math
import sys

class MhiConstructor:
    def __init__(self, background, threshold = 50, gaussian_blur_size = 31,):
        self.__th = threshold
        self.__gb_size = gaussian_blur_size
        if (background.ndim >= 3):
            background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        self.__background = cv.GaussianBlur(background, (self.__gb_size, self.__gb_size), 0)
        self.__diffs = []
        self.__mhi = None
        self.__mei = None

    def diff(self, foreground):
        if (foreground.ndim >= 3):
            foreground = cv.cvtColor(foreground, cv.COLOR_BGR2GRAY)
        foreground = cv.GaussianBlur(foreground, (self.__gb_size, self.__gb_size), 0)
        d = cv.absdiff(self.__background, foreground)
        _, d = cv.threshold(d, self.__th, 255, cv.THRESH_BINARY)

        # OPEN on diff
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        d = cv.morphologyEx(d, cv.MORPH_OPEN, kernel)

        self.__diffs.append(d)
        return d

    def construct_mhi(self, start_frame_index = 0, last_frame_index = 0):
        if len(self.__diffs) == 0:
            tb = sys.exc_info()[2]
            raise RuntimeError('list of history frames is empty').with_traceback(tb)

        if ((last_frame_index != 0 and last_frame_index >= len(self.__diffs)) \
            or (last_frame_index != 0 and last_frame_index < start_frame_index)):
            tb = sys.exc_info()[2]
            raise RuntimeError(f'last_frame_index is out of bound, max shall be ({start_frame_index}, {len(self.__diffs) - 1}]').with_traceback(tb)

        if last_frame_index == 0:
            last_frame_index = len(self.__diffs) - 1

        tao = last_frame_index - start_frame_index

        m = np.zeros_like(self.__diffs[0], dtype=int)
        for i in range(start_frame_index + 1, last_frame_index + 1):
            b = (self.__diffs[i] != 0)
            m_minus_1 = m - 1
            m_minus_1[m_minus_1 < 0] = 0
            m[b] = tao
            m[~b] = m_minus_1[~b]

        self.__mhi = cv.normalize(m.astype(np.uint8), None, 255, 0, norm_type=cv.NORM_MINMAX)

        return self.__mhi, tao

    def construct_mei(self):
        if self.__mhi is None:
            self.construct_mhi()

        self.__mei = np.zeros_like(self.__mhi, dtype = np.uint8)
        b = (self.__mhi != 0)
        self.__mei[b] = 255

        return self.__mei

    def compute_hu_moments(self):
        if (self.__mhi is None or self.__mei is None):
            raise RuntimeError('MHI and MEI has not been not constructed, call construct_mhi() and construct_mei() first')

        m = cv.moments(self.__mhi)
        mhi_hu = cv.HuMoments(m)

        m = cv.moments(self.__mei)
        mei_hu = cv.HuMoments(m)
        hu = np.concatenate((mhi_hu, mei_hu))

        # normalize hu to -1 * sign(hu) * log10(abs(hu))

        v1 = np.log10(np.absolute(hu))
        v2 = -1 * np.copysign(np.ones(hu.shape, dtype = np.float64), hu, dtype = np.float64)
        hu = v2 * v1

        return hu

    def compute_central_normalized_moments(self):
        if (self.__mhi is None or self.__mei is None):
            raise RuntimeError('MHI and MEI has not been not constructed, call construct_mhi() and construct_mei() first')

        mhi_mom = cv.moments(self.__mhi)
        mei_mom = cv.moments(self.__mei)
        mom = np.array([mhi_mom['nu20'], mhi_mom['nu11'], mhi_mom['nu02'], mhi_mom['nu30'], mhi_mom['nu21'], mhi_mom['nu12'], mhi_mom['nu03'], \
                        mei_mom['nu20'], mei_mom['nu11'], mei_mom['nu02'], mei_mom['nu30'], mei_mom['nu21'], mei_mom['nu12'], mei_mom['nu03']])

        return mom
