import requests
from bs4 import BeautifulSoup
import json
import cv2
import random
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import urllib.request


def get_genre_n_pages(genre, n=1):  # default is ==1 which means 50 entries
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "DNT": "1", "Connection": "close", "Upgrade-Insecure-Requests": "1", "Accept-Language": "en-US, en;q=0.5"}
    start = 'https://www.imdb.com/search/title/?title_type=feature&genres='
    mid = '&count='
    num = str(n * 50)
    genre_url = start + genre + mid + num
    print(genre_url)
    response = requests.get(genre_url, headers=headers)

    if not response.ok:
        print('Status Code:', response.status_code)
        raise Exception('Failed to get web page' + genre_url)

    doc = BeautifulSoup(response.text)

    return doc


def fetch_movie_page(imdb_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        }
    response = requests.get(imdb_url, headers=headers)
    if response.ok:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        return None


def get_genre(parsed_html):
    if parsed_html:
        genre_section = parsed_html.find('div', {'data-testid': 'genres'})
        if genre_section:
            genre_tags = genre_section.find_all('a')
            genres = [genre.text.strip() for genre in genre_tags]
            return genres
        else:
            return "Genres not found"
    else:
        return "Invalid HTML content"


def get_budget(movie_soup):
    try:
        budget_div = movie_soup.find('li', {"data-testid": "title-boxoffice-budget"})
        budget = budget_div.text.split("$")[1].split(" ")[0]
        budget = budget.replace(",", "")
        return int(budget)
    except (AttributeError, ValueError):
        return None


def get_trailer_link(movie_soup):
    try:
        official_trailer = movie_soup.find('a', {'aria-label': 'TrailerOfficial Trailer'})
        if official_trailer is None:
            link = movie_soup.find('script', {'type': 'application/json'})
            json_ob = json.loads(link.string)
            url = \
            json_ob['props']['pageProps']['aboveTheFoldData']['primaryVideos']['edges'][0]['node']['playbackURLs'][0][
                'url']
            return url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
        imdb_url = 'https://www.imdb.com' + official_trailer['href'].split('?')[0]
        response = requests.get(imdb_url, headers=headers)
        page = BeautifulSoup(response.text, 'html.parser')
        link = page.find('script', {'type': 'application/json'})
        json_ob = json.loads(link.string)
        videos = json_ob["props"]["pageProps"]["videoPlaybackData"]["video"]["playbackURLs"]
        url = videos[1]['url']
        urllib.request.urlretrieve(url, 'test.mp4')
        return url
    except (IndexError, ValueError):
        return None


LAST_FRAME_HIST = []


# TODO: MAKE THRESHOLD 0.02 BUT REMOVE VERY CLOSE SCENE CHANGES

def shot_detection_array_given(vid_array, threshold):
    lines = []
    shot_change = []
    for i in range(len(vid_array) - 1):  # Compare frame to the next one, so one less range
        frame = vid_array[i]
        next_frame = vid_array[i + 1]
        SD = same_shot(frame, next_frame, 10)  # TODO: PASS LAST ITERATIONS NEXT_FRAME CALC TO SD TO SAVE TIME
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


def get_shot_image_indices(vid_array, threshold):
    shot_changes = shot_detection_array_given(vid_array, threshold)

    images_i = []
    shot_start = 0  # TODO: Remove start and end of trailer cuz of title cards and production companies n stuff
    for i in range(len(shot_changes)):
        images_i.append(numpy.random.randint(shot_start, shot_changes[i]))
        shot_start = shot_changes[i]
    return shot_changes


def get_training_data_shot_frames():
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
              'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Short',
              'Sport', 'Thriller', 'War', 'Western']

    # Randomize selected genres
    # selected_genres = random.sample([_ for _ in range(len(genres))], 10)
    selected_genres = [10, 15, 12, 1, 19, 0, 3, 9, 6, 11]
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        fields = ['genres', 'budget', 'link']
        writer.writerow(fields)
        image_id = 0

        # Select movies from each genre
        for i in selected_genres:
            genre = genres[i]
            doc1 = get_genre_n_pages(genre=genre, n=1)
            movie_tags = doc1.find_all('li', class_='ipc-metadata-list-summary-item')

            found_movie = False
            j = 0
            text_file_lines = ["id, budget, genre"]
            while (not found_movie):
                imdb_url = 'https://www.imdb.com'
                a_tag = movie_tags[j].find('a')
                href_full = a_tag['href']
                href_part = href_full.split('?')[0]
                sample_url = imdb_url + href_part
                page = fetch_movie_page(sample_url)
                movie_genres = get_genre(page)
                movie_budget = get_budget(page)
                movie_trailer = get_trailer_link(page)
                if movie_budget != None and movie_trailer != None:
                    found_movie = True

                    frames = []
                    cap = cv2.VideoCapture(movie_trailer)
                    ret = True
                    while ret:
                        ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
                        if ret:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            frames.append(img)
                    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
                    image_indices = get_shot_image_indices(video, 0.05)
                    for index in image_indices:
                        image_id += 1
                        im = Image.fromarray(image_indeces[index])
                        im.save("images/%d.jpeg" % (image_id))  # Just uses an id number so no duplicates
                        text_file_lines.append("%d, %d, %s\n" % (image_id, movie_budget, genres[i]))

                j += 1
                print(movie_genres, movie_budget, movie_trailer, '\n')
        with open('image_to_movie_dict.txt', 'w') as f:
            f.writelines(text_file_lines)
    file.close()

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

a = get_training_data_shot_frames()
