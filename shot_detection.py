import numpy as np
import matplotlib.pyplot as plt
import cv2
import matlab
import numpy.linalg
from scipy.ndimage import correlate as corr
from scipy.ndimage import gaussian_filter as gaus

LAST_FRAME_HIST = []
#TODO: MAKE THRESHOLD 0.02 BUT REMOVE VERY CLOSE SCENE CHANGES

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
    lines = []
    for i in range(len(video) - 1): # Compare frame to the next one, so one less range
        frame = video[i]
        next_frame = video[i+1]
        SD = same_shot(frame, next_frame, 10) # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
        if SD > threshold:
            vis = np.concatenate((frame, next_frame), axis=1)
            plt.imshow(vis)
            plt.axis("off")
            plt.show()
            print(i, i+1)
            lines.append('%d, %d\n' % (i, i+1))

    with open('shot_changes_02.txt', 'a') as f:
        f.writelines(lines)

def shot_detection_array_given(vid_array, threshold):
    lines = []
    shot_change = []
    for i in range(len(video) - 1): # Compare frame to the next one, so one less range
        frame = vid_array[i]
        next_frame = vid_array[i+1]
        SD = same_shot(frame, next_frame, 10) # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
        if SD > threshold:
            shot_change.append(i + 1)

    return shot_change

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


def get_shot_image_i(vid_array, threshold):
    shot_changes = shot_detection_array_given(vid_array, threshold)

    images_i = []
    shot_start = 0 # TODO: Remove start and end of trailer cuz of title cards and production companies n stuff
    for i in range(len(shot_changes)):
        images_i.append(numpy.random.randint(shot_start, shot_changes[i]))
        shot_start = shot_changes[i]
    return shot_changes


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    shot_detection("video_name.mp4", 0.02)
