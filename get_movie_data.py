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

    doc = BeautifulSoup(response.text, 'html.parser')

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
        # Select the trailer at the top of the page
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


genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
          'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Short',
          'Sport', 'Thriller', 'War', 'Western']

# Randomize selected genres
# selected_genres = random.sample([_ for _ in range(len(genres))], 10)
selected_genres = [10, 15, 12, 1, 19, 0, 3, 9, 6, 11]
def get_training_data():
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        fields = ['genres', 'budget', 'link']
        writer.writerow(fields)

        # Select movies from each genre
        for i in selected_genres:
            genre = genres[i]
            doc1 = get_genre_n_pages(genre=genre, n=1)
            movie_tags = doc1.find_all('li', class_ = 'ipc-metadata-list-summary-item')

            found_movie = False
            j = 0
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
                    writer.writerow([movie_genres, movie_budget, movie_trailer])

                    # vidcap = cv2.VideoCapture(movie_trailer)
                    # success,image = vidcap.read()
                    # count = 0
                    # while success:
                    #     cv2.imwrite("test/movie%d_frame%d.jpg" % (i, count), image)     # save frame as JPEG file      
                    #     success,image = vidcap.read()
                    #     print('Read a new frame: ', success)
                    #     count += 1
                    #     success = False
                
                j += 1
                # print(movie_genres, movie_budget, movie_trailer, '\n')
    file.close()

if __name__ == '__main__':
    get_training_data()
