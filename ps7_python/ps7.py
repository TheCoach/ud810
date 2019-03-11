from mhi import MhiConstructor

import sys, traceback
import os.path
import cv2 as cv
import numpy as np
import re
from collections import namedtuple

ACTION_RANGES = {
    'A1P1T1': (29, 91), 'A1P1T2': (19, 77), 'A1P1T3': (27, 95),
    'A2P1T1': (24, 54), 'A2P1T2': (29, 61), 'A2P1T3': (33, 66),
    'A3P1T1': (25, 84), 'A3P1T2': (15, 85), 'A3P1T3': (18, 83),
    'A1P2T1': (15, 58), 'A1P2T2': (14, 54), 'A1P2T3': (15, 58),
    'A2P2T1': (20, 50), 'A2P2T2': (22, 53), 'A2P2T3': (17, 61),
    'A3P2T1': (12, 66), 'A3P2T2': (13, 73), 'A3P2T3': (15, 69),
    'A1P3T1': (20, 69), 'A1P3T2': (17, 66), 'A1P3T3': (16, 62),
    'A2P3T1': (25, 55), 'A2P3T2': (25, 53), 'A2P3T3': (26, 56),
    'A3P3T1': (12, 63), 'A3P3T2': (18, 77), 'A3P3T3': (14, 74)
}

# threshold of frame differecing
FRAME_DIFF_TH = {
    'P1': 70,
    'P2': 30,
    'P3': 50
}

# Gaussian blur kernel size when doing frame differencing
GB_SIZE = {
    'P1': 31,
    'P2': 21,
    'P3': 31
}

ActivityDescriptor = namedtuple('ActivityDescriptor', ['mhi', 'tao', 'mom'])

def run_mhi(vpath, frames_to_save, part_str, diff_threshold, gaussian_blur_size = 31, mhi_start_frame = 0, mhi_end_frame = 0):
    cap = cv.VideoCapture(vpath)
    ret, frame = cap.read()
    if (ret == False):
        raise RuntimeError(f'Failed to read frame from video file {vpath}.')

    grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mhi_ctor = MhiConstructor(grayF, diff_threshold, gaussian_blur_size)

    frame_n = 1
    frame_saving_i = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('No frame read. End of file?')
            break

        grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff = mhi_ctor.diff(grayF)

        # print(f'frame: #{frame_n}')
        # cv.imshow('diff', diff)
        # cv.imshow('frame', frame)

        if (frame_n in frames_to_save):
            cv.imwrite(os.path.join('output', 'ps7-' + part_str + '-' + str(frame_saving_i) + '.png'), diff)
            frame_saving_i += 1

        frame_n += 1

        # k = cv.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     break
        # else:
        #     continue

    cap.release()
    cv.destroyAllWindows()

    print(f'Total number of frames of \"{vpath}\": {frame_n}')
    mhi, tao = mhi_ctor.construct_mhi(mhi_start_frame, mhi_end_frame)
    mhi_ctor.construct_mei()
    # hu = mhi_ctor.compute_hu_moments()
    '''
     Hu[5], Hu[6] and Hu[7] seems to be very unstable in this case, especially the sign.
    '''
    mom = mhi_ctor.compute_central_normalized_moments()
    return ActivityDescriptor(mhi, tao, mom)

def train_data():
    data = {}
    for filename in os.listdir('input'):
        if filename.startswith('PS7') and filename.endswith('.avi'):
            seq = os.path.splitext(filename)[0].split('PS7')[1]
            _, p, a, t = re.split('[PAT]', seq)
            th = FRAME_DIFF_TH[seq[2:4]]
            gbsz = GB_SIZE[seq[2:4]]
            fstart = ACTION_RANGES[seq][0]
            fend = ACTION_RANGES[seq][1]
            data[seq.lower()] = run_mhi(os.path.join('input', filename), [], 'none', th, gbsz, fstart, fend)

    return data

def filter_seq(data, include_seq_name, exclude_seq_name = ' '):
    ret = {}
    for k, v in data.items():
        if (include_seq_name.lower() in k.lower() and exclude_seq_name.lower() not in k.lower()):
            ret[k] = v
    return ret

def filter_property(data, pname):
    if (pname is not None and pname.lower() not in ['mhi', 'tao', 'mom']):
        raise ValueError('pname shall be one of "mhi", "tao" or "mom"');
    ret = {}
    for k, v in data.items():
        ret[k] = getattr(v, pname)
    return ret

def compare_moments(h1, h2):
    # compute 2-Norm, i.e. distance between to vectors
    d = np.linalg.norm(h1 - h2)
    return d

'''
   3 x 3, r (row) is (action - 1), j (column) is (trial - 1)
'''
def build_confusion_matrix(dict_of_moments, mom):
    if (len(dict_of_moments) != 9):
        raise ValueError('dict_of_moments is not valid')

    ret = np.zeros((3,3), dtype = np.float64)
    for seq, m in dict_of_moments.items():
        _, a, p, t = re.split('[pat]', seq.lower())
        a, p, t = map(int, [a, p, t])
        ret[a - 1, t - 1] = compare_moments(m, mom)
    return ret

def evaluate(data):
    for p in range(1, 4): # each person
        for a in range(1, 4): # each action
            print('#############################################')
            print(f'Evaluate Action#{a} against Person#{p}')
            # get all action moments excluding Person-p
            action_mom = filter_property(filter_seq(data, 'a' + str(a), 'p' + str(p)), 'mom')
            person_mom = filter_property(filter_seq(data, 'p' + str(p)), 'mom')

            avg_mom = np.zeros((14,), dtype = np.float64)
            for m in action_mom.values():
                avg_mom += m
            avg_mom /= len(action_mom)

            cmat = build_confusion_matrix(person_mom, avg_mom)

            minindex = np.argmin(cmat, axis = 0)
            targetindex = np.array([a-1, a-1, a-1])
            if(not np.array_equal(minindex, targetindex)):
                print('XXX Evaluation failed XXX')
                print(cmat)
                print(minindex)
            else:
                print('YYY Evaluation passed YYY')
            print('#############################################\n')

def main(argv):
    try:
        data = train_data()
        evaluate(data)
    except RuntimeError as rte:
        print(f'MHI failed to run: {rte}')
        print(traceback.print_tb(rte.__traceback__))
    finally:
        cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
