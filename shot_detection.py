import numpy as np
import matplotlib.pyplot as plt
import cv2
import matlab
import numpy.linalg
from scipy.ndimage import correlate as corr
from scipy.ndimage import gaussian_filter as gaus

LAST_FRAME_HIST = []


def shot_detection(vid_name, threshold):
    frames = []

    cap = cv2.VideoCapture(vid_name)
    ret = True
    while ret:
        ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)

    min_sd = 10
    max_sd = -1
    for i in range(len(video) - 1): # Compare frame to the next one, so one less range
        frame = video[i]
        next_frame = video[i+1]
        SD = same_shot(frame, next_frame, 10) # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
        min_sd = min(SD, min_sd)
        max_sd = max(SD, max_sd)
        if SD > threshold and i < 3400:
            vis = np.concatenate((frame, next_frame), axis=1)
            plt.imshow(vis)
            plt.axis("off")
            plt.show()
            print(i, i+1)



def same_shot(img_1, img_2, size):
    global LAST_FRAME_HIST
    if len(LAST_FRAME_HIST) == 0:
        img_1_g = np.dot(img_1[..., :3], [0.299, 0.587, 0.114])
        img_2_g = np.dot(img_2[..., :3], [0.299, 0.587, 0.114])

        img_1_hist = intensity_hist(img_1_g, size)
        img_2_hist = intensity_hist(img_2_g, size)

        SD = 0
        for i in range(size):
            SD += abs(img_1_hist[i] - img_2_hist[i])

        height, width = len(img_2_g), len(img_2_g[0])
        SD = SD / (size * height * width)
        LAST_FRAME_HIST = img_2_hist
    else:
        img_2_g = np.dot(img_2[..., :3], [0.299, 0.587, 0.114])
        img_2_hist = intensity_hist(img_2_g, size)
        SD = 0
        for i in range(size):
            SD += abs(LAST_FRAME_HIST[i] - img_2_hist[i])
        height, width = len(img_2_g), len(img_2_g[0])
        SD = SD / (size * height * width)
        LAST_FRAME_HIST = img_2_hist
    return SD




def intensity_hist(img, size):
    dist = [0] * size
    max_i = 0
    min_i = 255
    for i in range(len(img)):
        for j in range(len(img[i])):
            dist_index = int(img[i][j] // (255.1 / size))
            dist[dist_index] += 1
            max_i = max(max_i, img[i][j])
            min_i = min(min_i, img[i][j])
    return dist


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    shot_detection("video_name.mp4", 0.05)
