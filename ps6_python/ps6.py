import utils
import pf

import cv2 as cv
import os
import sys
import math

def p1_a():
    cap = cv.VideoCapture(os.path.join('input', 'pres_debate.avi'))
    x, y, TEMP_W, TEMP_H = utils.get_template_position(os.path.join('input', 'pres_debate.txt'))
    cx = x + math.floor(TEMP_W / 2)
    cy = y + math.floor(TEMP_H / 2)

    ret, frame = cap.read()
    grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    t = grayF[y : y + TEMP_H, x : x + TEMP_W] # template
    cv.imwrite(os.path.join('output', 'ps6-1-a-1.png'), t)

    partf = pf.ParticleFilter(cx, cy, t, mse_sigma=10, num_particles=200, dynamic_sigma=30)

    frame_t = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('No frame read')
            break

        grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        partf.filter(grayF)
        partf.visualize(frame)

        cv.imshow('frame', frame)

        if (frame_t == 28):
            cv.imwrite(os.path.join('output', 'ps6-1-a-2.png'), frame)
        if (frame_t == 84):
            cv.imwrite(os.path.join('output', 'ps6-1-a-3.png'), frame)
        if (frame_t == 144):
            cv.imwrite(os.path.join('output', 'ps6-1-a-4.png'), frame)

        frame_t += 1
        k = cv.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        else:
            continue

    cv.destroyAllWindows()
    cap.release()

def p1_e():
    cap = cv.VideoCapture(os.path.join('input', 'noisy_debate.avi'))
    x, y, TEMP_W, TEMP_H = utils.get_template_position(os.path.join('input', 'noisy_debate.txt'))
    cx = x + math.floor(TEMP_W / 2)
    cy = y + math.floor(TEMP_H / 2)

    ret, frame = cap.read()
    grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    t = grayF[y : y + TEMP_H, x : x + TEMP_W] # template

    partf = pf.ParticleFilter(cx, cy, t, mse_sigma=5, num_particles=200, dynamic_sigma=30)

    frame_t = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('No frame read')
            break

        grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        partf.filter(grayF)
        partf.visualize(frame)
        cv.imshow('frame', frame)

        if (frame_t == 14):
            cv.imwrite(os.path.join('output', 'ps6-1-e-1.png'), frame)
        if (frame_t == 32):
            cv.imwrite(os.path.join('output', 'ps6-1-e-2.png'), frame)
        if (frame_t == 46):
            cv.imwrite(os.path.join('output', 'ps6-1-e-3.png'), frame)

        frame_t += 1
        k = cv.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        else:
            continue

    cv.destroyAllWindows()
    cap.release()

def p2_a():
    cap = cv.VideoCapture(os.path.join('input', 'pres_debate.avi'))
    x, y, TEMP_W, TEMP_H = utils.get_template_position(os.path.join('input', 'pres_debate_hand.txt'))
    cx = x + math.floor(TEMP_W / 2)
    cy = y + math.floor(TEMP_H / 2)

    ret, frame = cap.read()
    grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    t = grayF[y : y + TEMP_H, x : x + TEMP_W] # template

    partf = pf.ParticleFilter(cx, cy, t, mse_sigma=3, num_particles=3000, dynamic_sigma=150)

    frame_t = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('No frame read')
            break

        grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        partf.filter(grayF, 0.1)
        partf.visualize(frame)

        cv.imshow('frame', frame)

        if (frame_t == 15):
            cv.imwrite(os.path.join('output', 'ps6-2-a-1.png'), frame)
        if (frame_t == 54):
            cv.imwrite(os.path.join('output', 'ps6-2-a-2.png'), frame)
        if (frame_t == 140):
            cv.imwrite(os.path.join('output', 'ps6-2-a-3.png'), frame)

        frame_t += 1
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        else:
            continue

    cv.destroyAllWindows()
    cap.release()

def p2_b():
    cap = cv.VideoCapture(os.path.join('input', 'noisy_debate.avi'))
    x, y, TEMP_W, TEMP_H = utils.get_template_position(os.path.join('input', 'pres_debate_hand.txt'))
    cx = x + math.floor(TEMP_W / 2)
    cy = y + math.floor(TEMP_H / 2)

    ret, frame = cap.read()
    grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    t = grayF[y : y + TEMP_H, x : x + TEMP_W] # template

    partf = pf.ParticleFilter(cx, cy, t, mse_sigma=4, num_particles=4000, dynamic_sigma=400)

    frame_t = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('No frame read')
            break

        grayF = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        partf.filter(grayF, 0.3)
        partf.visualize(frame, verbose=2)

        cv.imshow('frame', frame)

        if (frame_t == 15):
            cv.imwrite(os.path.join('output', 'ps6-2-b-1.png'), frame)
        if (frame_t == 54):
            cv.imwrite(os.path.join('output', 'ps6-2-b-2.png'), frame)
        if (frame_t == 140):
            cv.imwrite(os.path.join('output', 'ps6-2-b-3.png'), frame)

        frame_t += 1
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        else:
            continue

    cv.destroyAllWindows()
    cap.release()

def main(argv):
    # p1_a()
    # p1_e()
    p2_a()

if __name__ == '__main__':
    main(sys.argv[1:])
