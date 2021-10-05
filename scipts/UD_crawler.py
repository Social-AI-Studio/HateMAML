import requests
from bs4 import BeautifulSoup
import sys

if sys.version < '3':
    from urllib2 import urlopen
else:
    from urllib.parse import quote as urlquote


class urban_dictonary():

    # Urban_dictonary crawler to get meaning

    def __init__(self):
        self.url = 'http://www.urbandictionary.com'

    def get_meaning(self, term, till_page=1):

        """
        Given meanings upto a certain page

        INPUT:
          term : str for which you want the meaning
          till_page: int till which page you want all the meanings from

        RETURN:

        A list containg dictionaries of{
        Word: str root word of that perticular post (word may or may not be equal to term)
        Meaning: str meaning of the root word of that post
        Link: str string leading to other UD pages
        Example: str word used in a sentence
        Upvote: int
        Downvote: int


        """

        cur_page = 1
        page_url = self.url + '/define.php?term=' + urlquote(term) + '&page=' + str(cur_page)
        meanings = []
        while (cur_page <= till_page):

            page_url = self.url + '/define.php?term=' + urlquote(term) + '&page=' + str(cur_page)
            page = requests.get(page_url)

            page_soup = BeautifulSoup(page.text, 'html.parser')

            for panel in page_soup.find_all('div', class_='def-panel'):
                word = panel.find('a', class_='word')
                meaning = panel.find('div', class_='meaning')
                hyper_links = meaning.find_all('a', class_='autolink')
                example = panel.find('div', class_='example')
                upvote = panel.find('div', class_='def-footer').find('a', class_='up')
                downvote = panel.find('div', class_='def-footer').find('a', class_='down')

                meanings.append({
                    'Word': word.text,
                    'Meaning': meaning.text,
                    'Link': [link['href'] for link in hyper_links if link.has_attr('href')],
                    'Example': example.text,
                }
                )
            cur_page += 1
        return meanings