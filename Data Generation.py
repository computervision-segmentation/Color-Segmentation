import os
import cv2
import numpy as np

train_folder = "Proj1_Train"
test_folder = "Proj1_Train_additional"
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def im_adjust(im):
    # convert RGB image into LAB
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    # perform clahe histogram balancing on the L channel
    im_lab[:, :, 0] = clahe.apply(im_lab[:, :, 0])
    # convert LAB image back into RGB
    im_adj = cv2.cvtColor(im_lab, cv2.COLOR_Lab2BGR)

    return im_adj


def crop(event, x, y, flag, param):
    global pts

    # if the left mouse button was clicked, record the starting (x, y) coordinate and indicate that cropping
    # is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if pts.size == 0:
            new_pt = np.array([[x, y]], np.int32)
            pts = np.append(pts, new_pt)
            cv2.circle(img_m, (x, y), 1, (0, 0, 255), -1)

        else:
            cv2.circle(img_m, (x, y), 1, (0, 0, 255), -1)
            cv2.line(img_m, (pts[-2], pts[-1]), (x, y), (255, 255, 0), 1)
            new_pt = np.array([[x, y]], np.int32)
            pts = np.append(pts, new_pt)


def end_mouse(event, x, y, flag, param):
    pass

red_barrel = []
red_other = []
ground = []
sky = []
for filename in os.listdir(train_folder):
    # read one test image
    img = cv2.imread(os.path.join(train_folder, filename), -1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_m = im_adjust(img)  # use adjusted image for better visualization. training data are not adjusted.
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', crop)

    # Store pixels of red barrel
    pts = np.array([], np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    while True:
        cv2.imshow('image', img_m)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.setMouseCallback('image', end_mouse)
            break

    if pts.size != 0:
        pts = pts.reshape((-1, 1, 2))
        img_m = cv2.polylines(img_m, [pts], True, (0, 0, 255), 1)
        cv2.fillPoly(mask, [pts], 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img_rb = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('image', img_rb)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_rb[i, j, 0] != 0 and img_rb[i, j, 1] != 0 and img_rb[i, j, 2] != 0:
                    red_barrel.append([img_hsv[i, j, 0], img_hsv[i, j, 1], img_hsv[i, j, 2]])
        cv2.waitKey(0) & 0xFF
# save it for future use
red_barrel = np.array(red_barrel, dtype=np.uint8)
np.savetxt('red_barrel.tar.gz', red_barrel)

for filename in os.listdir(train_folder):
    # read one test image
    img = cv2.imread(os.path.join(train_folder, filename), -1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_m = im_adjust(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', crop)

    # Store pixels of other red objects
    pts = np.array([], np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    while True:
        cv2.imshow('image', img_m)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.setMouseCallback('image', end_mouse)
            break

    if pts.size != 0:
        pts = pts.reshape((-1, 1, 2))
        img_m = cv2.polylines(img_m, [pts], True, (0, 0, 255), 1)
        cv2.fillPoly(mask, [pts], 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img_rb = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('image', img_rb)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_rb[i, j, 0] != 0 and img_rb[i, j, 1] != 0 and img_rb[i, j, 2] != 0:
                    red_other.append([img_hsv[i, j, 0], img_hsv[i, j, 1], img_hsv[i, j, 2]])
        cv2.waitKey(0) & 0xFF
# save it for future use
red_other = np.array(red_other, dtype=np.uint8)
np.savetxt('red_other.tar.gz', red_other)

for filename in os.listdir(train_folder):
    # read one test image
    img = cv2.imread(os.path.join(train_folder, filename), -1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_m = im_adjust(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', crop)

    # Store pixels of sky
    pts = np.array([], np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    while True:
        cv2.imshow('image', img_m)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.setMouseCallback('image', end_mouse)
            break

    if pts.size != 0:
        pts = pts.reshape((-1, 1, 2))
        img_m = cv2.polylines(img_m, [pts], True, (0, 0, 255), 1)
        cv2.fillPoly(mask, [pts], 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img_rb = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('image', img_rb)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_rb[i, j, 0] != 0 and img_rb[i, j, 1] != 0 and img_rb[i, j, 2] != 0:
                    sky.append([img_hsv[i, j, 0], img_hsv[i, j, 1], img_hsv[i, j, 2]])
        cv2.waitKey(0) & 0xFF
# save it for future use
sky = np.array(sky, dtype=np.uint8)
np.savetxt('sky.tar.gz', sky)

for filename in os.listdir(train_folder):
    # read one test image
    img = cv2.imread(os.path.join(train_folder, filename), -1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_m = im_adjust(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', crop)

    # Store pixels of ground
    pts = np.array([], np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    while True:
        cv2.imshow('image', img_m)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.setMouseCallback('image', end_mouse)
            break

    if pts.size != 0:
        pts = pts.reshape((-1, 1, 2))
        img_m = cv2.polylines(img_m, [pts], True, (0, 0, 255), 1)
        cv2.fillPoly(mask, [pts], 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img_rb = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('image', img_rb)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_rb[i, j, 0] != 0 and img_rb[i, j, 1] != 0 and img_rb[i, j, 2] != 0:
                    ground.append([img_hsv[i, j, 0], img_hsv[i, j, 1], img_hsv[i, j, 2]])
        cv2.waitKey(0) & 0xFF
# save it for future use
ground = np.array(ground, dtype=np.uint8)
np.savetxt('ground.tar.gz', ground)
