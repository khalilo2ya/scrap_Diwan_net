#https://www.aldiwan.net/poem101879.html
from bs4 import BeautifulSoup as bs
import requests

base_url_poet = "https://www.aldiwan.net/poem101878.html"

r = requests.get(base_url_poet)

soup = bs(r.content, 'lxml')
contents = soup.prettify()
poem = soup.select_one("#poem_content").get_text("\n", strip=True )
print(poem)
