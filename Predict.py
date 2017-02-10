import os
import sys
import cv2
import numpy as np

# location of test images
test_folder = "Proj1_Test"

red_barrel_param = np.load('red_barrel_param.npy')
red_other_param = np.load('red_other_param.npy')
sky_param = np.load('sky_param.npy')
ground_param = np.load('ground_param.npy')


def gauss(x, m, s):
    for i in range(len(s)):
        if s[i, i] <= sys.float_info[3]:
            s[i, i] = sys.float_info[3]
    s_inv = np.linalg.inv(s)
    xm = np.matrix(x - m)
    return (2.0 * np.pi) ** (-len(x[1]) / 2.0) * (1.0 / (np.linalg.det(s) ** 0.5)) * \
           np.exp(-0.5 * np.sum(np.multiply(xm * s_inv, xm), axis=1))


def detect_barrel(image):
    # intrinsic parameters of the camera
    K = np.matrix([[413.8794311523438, 0.0, 667.1858329372201], [0.0, 415.8518917846679, 500.6681745269121],
                   [0.0, 0.0, 1.0]])
    # convert the image into HSV space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # calculate the probability of pixels being in each color. the one with the highest probability is the estimated
    # color
    p = np.ndarray([img_hsv.shape[0] * img_hsv.shape[1], 4], np.float64)
    img_hsv_vec = img_hsv.reshape((img_hsv.shape[0] * img_hsv.shape[1], 3))
    for c in range(4):
        p[:, 0: 1] += gauss(img_hsv_vec, red_barrel_param[c]['mu'],
                            red_barrel_param[c]['sigma']) \
                      * red_barrel_param[c]['prob']
    for c in range(6):
        p[:, 1: 2] += gauss(img_hsv_vec, red_other_param[c]['mu'],
                            red_other_param[c]['sigma']) \
                      * red_other_param[c]['prob']
        p[:, 2: 3] += gauss(img_hsv_vec, sky_param[c]['mu'],
                            sky_param[c]['sigma']) \
                      * sky_param[c]['prob']
        p[:, 3: 4] += gauss(img_hsv_vec, ground_param[c]['mu'],
                            ground_param[c]['sigma']) \
                      * ground_param[c]['prob']
    label = np.argmax(p, axis=1) + 2
    label = label.astype(np.uint8)
    label.resize((img_hsv.shape[0], img_hsv.shape[1]))

    # extract preliminary red barrel mask
    mask_red_barrel = cv2.inRange(label, 2, 2)
    # cv2.imwrite('initial_mask.png', mask_red_barrel)

    # remove noise, then dilate to get the possible barrel regions
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask_red_barrel, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imwrite('opening.png', opening)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # cv2.imwrite('sure_bg.png', sure_bg)

    # remove the regions that are clearly not rectangular shape
    _, contours, _ = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    mask = np.zeros((img_hsv.shape[0], img_hsv.shape[1]), np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(areas)
        _, _, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area
        if extent > 0.7:
            pts = contour.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

    # cv2.imwrite('mask2.png', mask)
    # dilate again to merge neighboring regions
    mask = cv2.dilate(mask, kernel, iterations=3)
    # cv2.imwrite('mask3.png', mask)

    # choose the regions with proper height/width ratio as barrels
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    bl_x, bl_y, tr_x, tr_y, dist = [], [], [], [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(areas)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area
        if extent > 0.8 and 1.15 < h / w < 2:
            bl_x.append(int(x))
            bl_y.append(int(y + h))
            tr_x.append(int(x + w))
            tr_y.append(int(y))
            pts = np.array([[x - w / 2, y - h / 2, 1], [x - w / 2, y + h / 2, 1], [x + w / 2, y - h / 2, 1],
                            [x + w / 2, y + h / 2, 1]])
            pos = np.linalg.inv(K) * pts.T
            # calculate the distance with camera parameters
            distance = (0.93 / np.linalg.norm(pos[:, 0] - pos[:, 1]) + 0.93 / np.linalg.norm(pos[:, 2] - pos[:, 3]) \
                        + 0.53 / np.linalg.norm(pos[:, 1] - pos[:, 2]) + 0.53 / np.linalg.norm(pos[:, 1] - pos[:, 2]))\
                        / 4.0
            dist.append(distance)

    return bl_x, bl_y, tr_x, tr_y, dist

for filename in os.listdir(test_folder):
    # read one test image
    img = cv2.imread(os.path.join(test_folder, filename), -1)
    bl_x, bl_y, tr_x, tr_y, dist = detect_barrel(img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(bl_x)):
        img = cv2.rectangle(img, (bl_x[i], bl_y[i]), (tr_x[i], tr_y[i]), (0, 255, 0), 2)
        cv2.putText(img, 'd = {0}m'.format(str(dist[i])), (bl_x[i], bl_y[i]+25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(filename, img)
    cv2.imshow('image', img)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
