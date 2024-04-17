import requests
from bs4 import BeautifulSoup
import json
import random
import csv
import pandas as pd

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

def get_name(movie_soup):
    if movie_soup:
        name_section = movie_soup.find('h1', {'data-testid': 'hero__pageTitle'})
        name = name_section.text.strip()
        print(name)
        return name
    else:
        return "Invalid HTML content"


def get_genre(movie_soup):
    if movie_soup:
        genre_section = movie_soup.find('div', {'data-testid': 'genres'})
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
        budget = budget_div.text.split(" ")[0][7:]
        print(budget)
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
        return url
    except (IndexError, ValueError):
        return None

def get_movie_data(num_of_movies):
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
          'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Sport', 'Thriller', 'War', 'Western']
    titles = []

    with open('imdb_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        fields = ['name', 'genres', 'budget', 'link']
        writer.writerow(fields)

        # Select movies from any genre
        for _ in range(num_of_movies):
            genre = random.choice(genres)
            doc1 = get_genre_n_pages(genre=genre, n=2)
            movie_i = list(range(0, 100))
            movie_tags = doc1.find_all('li', class_ = 'ipc-metadata-list-summary-item')

            found_movie = False
            while (not found_movie):
                j = random.choice(movie_i)
                imdb_url = 'https://www.imdb.com'
                a_tag = movie_tags[j].find('a')
                href_full = a_tag['href']
                href_part = href_full.split('?')[0]
                sample_url = imdb_url + href_part
                page = fetch_movie_page(sample_url)
                movie_title = get_name(page)
                movie_genres = get_genre(page)
                movie_budget = get_budget(page)
                movie_trailer = get_trailer_link(page)
                if movie_budget != None and movie_trailer != None and movie_title not in titles:
                    found_movie = True
                    titles.append(movie_title)
                    writer.writerow([movie_title, movie_genres, movie_budget, movie_trailer])
    file.close()

if __name__ == '__main__':
    get_movie_data(num_of_movies=30)

    file = pd.read_csv('imdb_data.csv')
    train = file.sample(frac=0.9)
    test = file.loc[~file.index.isin(train.index)]

    train.to_csv("train_movies.csv", index=False)
    test.to_csv("test_movies.csv", index=False)


    
