from bs4 import BeautifulSoup as bs
from numpy import integer
import requests
import json
base_url_poet = "https://www.aldiwan.net/cat-poets-tunisia"
base_path = "https://www.aldiwan.net/"


def get_poem(url, poet_name, poem_name):
    r = requests.get(url)
    soup = bs(r.content, 'lxml')
    contents = soup.prettify()
    poem = soup.select_one("#poem_content").get_text("\n", strip=True )
    print(poem)
    poem_dict = {'poet_name': poet_name, 'poem_name': poem_name, 'poem_content': poem}
    poems.append(poem_dict)


def get_info_poet(url, poet_name):
    r = requests.get(url)
    soup = bs(r.content, 'lxml')
    contents = soup.prettify()
    poems = soup.select("div.col-sm-12.col-md a")
    for index, poem in enumerate(poems):
        relative_path = poem['href']
        full_path = base_path + relative_path
        print(full_path)
        poem_name = poem.get_text(" ", strip=True )
        print(poet_name, poem_name)
        get_poem(full_path, poet_name, poem_name)



r = requests.get(base_url_poet)

soup = bs(r.content, 'lxml')
contents = soup.prettify()
poets = soup.select("div.col-6.col-md-8 a")
poems=[]
for index, poet in enumerate(poets):
    relative_path = poet['href']
    full_path = base_path + relative_path
    poet_name = poet.get_text(" ", strip=True )
    # print(poet_name)
    get_info_poet(full_path, poet_name)


with open('poems.json', 'w') as f:
    json.dump(poems, f)