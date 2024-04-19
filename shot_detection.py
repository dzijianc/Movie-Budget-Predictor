import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import ast
import random

LAST_FRAME_HIST = []
#TODO: MAKE THRESHOLD 0.02 BUT REMOVE VERY CLOSE SCENE CHANGES
def intensity_hist(img, size):
    dist2 = [0] * size
    dist_index2 = (img // (255.1 / size)).astype(int).flatten()
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
    elif 1_000_000 <= budget < 10_000_000:
        return 1
    elif 10_000_000 <= budget < 50_000_000:
        return 2
    elif 50_000_000 <= budget < 100_000_000:
        return 3
    else:
        return 4

def get_genre_label(genres):
    all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
                'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Sport', 'Thriller', 'War', 'Western']
    
    ret = [0] * len(all_genres)
    for genre in genres:
        i = all_genres.index(genre)
        ret[i] = 1
    return ret

def shot_detection(csvfile, save_to, image_dir, threshold):
    file = pd.read_csv(csvfile)
    movie_budgets = file['budget']
    movie_links = file['link']
    movie_genres = file['genres']
    data = {'filename': [], 'budget': [], 'genres': []}

    for i in range(len(movie_links)):
        frames = []
        budget = get_budget_label(movie_budgets[i])
        link = movie_links[i]
        genres = get_genre_label(ast.literal_eval(movie_genres[i]))

        shot_end = 0
        # Get all the movie frames
        link = movie_links[i]
        vidcap = cv2.VideoCapture(link)
        success,image = vidcap.read()
        while success:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success,image = vidcap.read()

        print("All movie frames read.")

        shots = []
        last_frame = -1
        shot_start = 0
        for j in range(len(frames) - 1):
            cur_frame = frames[j]
            next_frame = frames[j + 1]
            SD = same_shot(cur_frame, next_frame, 10) # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
            if SD > threshold:
                if (last_frame == -1 or j - last_frame > 10):
                    shot_end = j + 1
                    shots.append((shot_start, shot_end))
                    shot_start = shot_end
                last_frame = j

        # Select up to 15 shots at random per movie
        if len(shots) > 15:
            shots_sample = random.sample(shots, 15)
        else:
            shots_sample = random.sample(shots, len(shots))
        for shot_range in shots_sample:
            idx = np.random.randint(shot_range[0], shot_range[1])
            selected_shot = frames[idx][...,::-1]
            filename = image_dir + "movie%d_frame%d.jpg" % (i + 81, idx)
            cv2.imwrite(filename, selected_shot)

            data['filename'].append(filename)
            data['budget'].append(budget)
            data['genres'].append(genres)
        
    df = pd.DataFrame(data)
    df.to_csv(save_to, index=False, mode='w')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    shot_detection("train_movies.csv", "train_set_15_shots.csv", "train_frames_15_shots/", 0.02)
    shot_detection("test_movies.csv", "test_set_15_shots.csv", "test_frames_15_shots/", 0.02)
    shot_detection("val_movies.csv", "val_set.csv", "val_frames/", 0.02)