import numpy as np
import cv2
import pandas as pd
import ast
import random

LAST_FRAME_HIST = []
def intensity_hist(img, size):
    """Calculate the intesity histogram for a given greyscale image
    Params
    ------
    img (np array): A grayscale image

    size (int): The number of equally sized buckets to be created for the histogram (e.g. if size = 2, the histogram
    will just have two buckets, one for intensity less than 50% and one for intensity of 50% or more)

    Returns
    ------
    hist (array): The histogram for the image, represented as a 1 x size int array where hist[i] is the number of
    pixels in img with that buckets intesnity range
    """
    dist2 = [0] * size
    dist_index2 = (img // (255.1 / size)).astype(int).flatten()
    bc = np.bincount(dist_index2)
    dist2[:len(bc)] += bc
    return dist2


def same_shot(img_1, img_2, size):
    """Generate the intensity histograms for two images and compute a score representing the difference in the two for
    the sake of shot detection
    Params
    ------
    img_1 (np array): A colored image, corresponding to a frame in a movie trailer

    img_2 (np array): A colored image, corresponding to the next frame in the movie trailer

    size (int): The number of equally sized buckets to be created for the histogram (e.g. if size = 2, the histogram
    will just have two buckets, one for intensity less than 50% and one for intensity of 50% or more)

    LAST_FRAME_HIST (list): A global variable storing the intensity histogram for img_2 from the last same_shot call.
    If this list is not empty, then this is used as the histogram for img_1, since this function is only called in
    same_shot, where we incremently compare each adjacent frame pairs for a trailer (i.e. first compares frame 0 and
    frame 1, then frame 1 and frame 2, and so frame 1 is img_2 in the first call and img_1 in the second call)

    Returns
    ------
    SD (int): A sum of the absolute differences of the intensity histograms for img_1 and img_2. This score is
    normalized by dividing by size and the number of pixels in one of the images.
    """
    # For the sake of performance,
    global LAST_FRAME_HIST
    if len(LAST_FRAME_HIST) == 0:
        img_1_g = np.dot(img_1[..., :3], [0.299, 0.587, 0.114])
        img_2_g = np.dot(img_2[..., :3], [0.299, 0.587, 0.114])

        img_1_hist = intensity_hist(img_1_g, size)
        img_2_hist = intensity_hist(img_2_g, size)

        SD = 0
        for i in range(size):
            SD += abs(img_1_hist[i] - img_2_hist[i])

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
    """Gets the label group for a given budget

    Params
    ------
    budget (int): Budget of the movie

    Returns
    ------
    label (int): The interval group the budget amount falls under
    """
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
    """Gets a list denoting what genres the movie was tagged with 

    Params
    ------
    genres (List): A list of genres where each item has to be one of the 22 defined genres:
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'

    Returns
    ------
    ret (List): 22-length list where each item is either 0 if the movie does not belong to that genre
    or 1 if the movie does belong to that genre
    """
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
    """Given a csvfile with a list of movies, detects shots from their trailer and selects one frame per shot for the
    sake of generating model training data, saving each frame in the specified image directory as well as saving a csv
    that ties each image saved to the movie it came from

    Params
    ------
    csvfile (CSV): A csv file containing a list of movies where each row
    has 4 pieces of data in the order: 
    
    name (str): name of the movie
    genres (str): str representation of a List of genres for the movie 
    budget (int): budget of the movie 
    link (str): Link to an mp4 file containing the movie trailer on IMDb

    save_to (str): The CSV file created where each row contains the name of a saved movie frame
    and the associated budget and genres of the movie

    image_dir (str): The directory to save the selected frames in JPG of a movie

    threshold (float): The threshold for shot detection
    """
    global LAST_FRAME_HIST
    file = pd.read_csv(csvfile)
    movie_budgets = file['budget']
    movie_links = file['link']
    movie_genres = file['genres']
    data = {'filename': [], 'budget': [], 'genres': []}

    for i in range(len(movie_links)):
        frames = []
        budget = get_budget_label(movie_budgets[i])
        genres = get_genre_label(ast.literal_eval(movie_genres[i]))

        # Get all the movie frames from trailer
        link = movie_links[i]
        vidcap = cv2.VideoCapture(link)
        success,image = vidcap.read()
        while success:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success,image = vidcap.read()

        print("All movie frames read.")

        LAST_FRAME_HIST = []
        shots = []
        last_frame = -1
        shot_start = 0
        for j in range(len(frames) - 1):
            cur_frame = frames[j]
            next_frame = frames[j + 1]
            SD = same_shot(cur_frame, next_frame, 10)
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