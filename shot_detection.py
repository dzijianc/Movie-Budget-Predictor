import numpy as np
import matplotlib.pyplot as plt
import cv2
import matlab
import numpy.linalg
from scipy.ndimage import correlate as corr
from scipy.ndimage import gaussian_filter as gaus
import pandas as pd

LAST_FRAME_HIST = []
#TODO: MAKE THRESHOLD 0.02 BUT REMOVE VERY CLOSE SCENE CHANGES
def intensity_hist(img, size):
    # print(img.shape)
    # print(len(img))
    # dist = [0] * size
    # max_i = 0
    # min_i = 255
    # for i in range(len(img)):
    #     for j in range(len(img[i])):
    #         dist_index = int(img[i][j] // (255.1 / size))
    #         dist[dist_index] += 1
    #         max_i = max(max_i, img[i][j])
    #         min_i = min(min_i, img[i][j])

    dist2 = [0] * size
    dist_index2 = (img // (255.1 / size)).astype(int).flatten()
    # print(dist_index2.shape)
    bc = np.bincount(dist_index2)
    dist2[:len(bc)] += bc
    return dist2


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

def get_budget_label(budget):
    if budget < 1_000_000:
        return 0
    elif 1_000_000 <= budget <= 10_000_000:
        return 1
    elif 10_000_000 <= budget <= 25_000_000:
        return 2
    elif 25_000_000 <= budget <= 50_000_000:
        return 3
    else:
        return 4


def shot_detection(csvfile, threshold):
    file = pd.read_csv(csvfile)
    movie_budgets = file['budget']
    movie_links = file['link']
    data = {'filename': [], 'budget': []}

    for i in range(len(movie_links)):
        frames = []
        budget = movie_budgets[i]
        link = movie_links[i]
        shot_end = 0
        # Get all the movie frames
        link = movie_links[i]
        vidcap = cv2.VideoCapture(link)
        success,image = vidcap.read()
        while success:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success,image = vidcap.read()

        print("Movie frames saved")

        shot_start = 0
        for j in range(len(frames) - 1):
            cur_frame = frames[j]
            next_frame = frames[j + 1]
            SD = same_shot(cur_frame, next_frame, 10) # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
            if SD > threshold:
                shot_end = j + 1
                idx = np.random.randint(shot_start, shot_end)
                selected_shot = frames[idx]
                filename = "test2/movie%d_frame%d.jpg" % (i, idx)
                cv2.imwrite(filename, selected_shot)
                shot_start = shot_end

                data['filename'].append(filename)
                data['budget'].append(get_budget_label(budget))
        
    df = pd.DataFrame(data)
    df.to_csv("test2.csv", index=False, mode='w')

# def shot_detection(vid_link, threshold):
#     frames = []

#     cap = cv2.VideoCapture(vid_link)
#     ret = True
#     while ret:
#         ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
#         if ret:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             frames.append(img)
#     video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
#     lines = []
#     for i in range(len(video) - 1): # Compare frame to the next one, so one less range
#         frame = video[i]
#         next_frame = video[i+1]
#         SD = same_shot(frame, next_frame, 10) # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
#         if SD > threshold:
#             vis = np.concatenate((frame, next_frame), axis=1)
#             plt.imshow(vis)
#             plt.axis("off")
#             plt.show()
#             print(i, i+1)
#             lines.append('%d, %d\n' % (i, i+1))

#     with open('shot_changes_02.txt', 'a') as f:
#         f.writelines(lines)


# def shot_detection_array_given(vid_array, threshold):
#     lines = []
#     shot_change = []
#     for i in range(len(vid_array) - 1):  # Compare frame to the next one, so one less range
#         frame = vid_array[i]
#         next_frame = vid_array[i + 1]
#         SD = same_shot(frame, next_frame, 10)  # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
#         if SD > threshold:
#             shot_change.append(i + 1)

#     return shot_change


# def get_shot_image_indices(vid_array, threshold):
#     shot_changes = shot_detection_array_given(vid_array, threshold)

#     images_i = []
#     shot_start = 0  # TODO: Remove start and end of trailer cuz of title cards and production companies n stuff
#     for i in range(len(shot_changes)):
#         images_i.append(np.random.randint(shot_start, shot_changes[i]))
#         shot_start = shot_changes[i]
#     return shot_changes


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    shot_detection("test.csv", 0.02)
