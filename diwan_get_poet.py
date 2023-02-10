#https://www.aldiwan.net/cat-poet-Abu-Bakr-al-Tunisi
from bs4 import BeautifulSoup as bs
import requests

base_url_poet = "https://www.aldiwan.net/cat-poet-Abu-Bakr-al-Tunisi"
base_path = "https://www.aldiwan.net/"
r = requests.get(base_url_poet)

soup = bs(r.content, 'lxml')
contents = soup.prettify()
poems = soup.select("div.col-sm-12.col-md a")

for index, poem in enumerate(poems):
    relative_path = poem['href']
    full_path = base_path + relative_path
    print(full_path)