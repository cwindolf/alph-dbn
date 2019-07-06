from bs4 import BeautifulSoup
from functools import partial
from time import sleep
import re, requests
soup = partial(BeautifulSoup, features='html5lib')
get = partial(requests.get, timeout=10)

WIKI = 'http://talkinwhipapedia.wikia.com'
OUT = 'lpc.txt'

s = soup(get(WIKI).text)
albums = s.find(id="Albums").parent.next_sibling.next_sibling.find_all('a')

call_urls = []
for album in albums:
    sleep(10)
    s = soup(get(WIKI + album['href']).text)
    calls = s.find(id="WikiaArticle").find_all('a')
    for call in calls:
        if 'redlink' not in call['href']:
            href = call['href'][len(WIKI):] if 'http://' in call['href'] else call['href']
            print('Might add:', href)
            if '/wiki/' in href:
                call_urls.append(WIKI + href)

with open(OUT, 'w') as out:
    new_call_urls = []
    while call_urls:
        for url in call_urls:
            sleep(10)
            try:
                call = soup(get(url).text).find(id="mw-content-text")
                for line in call.stripped_strings:
                    if (re.search('[a-zA-Z]', line)
                        and 'http' not in line
                        and 'retrieved from' not in line):
                        out.write(' '.join(line.split()) + '\n')
            except:
                print('Failed to get call:', url)
                new_call_urls.append(url)
        call_urls = new_call_urls
        new_call_urls = []

