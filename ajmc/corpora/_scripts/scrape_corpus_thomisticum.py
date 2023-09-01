import requests
from bs4 import BeautifulSoup

url = "https://www.corpusthomisticum.org/iopera.html"
fichier_destination = "/mnt/ajmcdata1/data/corpus_thomisticum/data/corpus.txt"


def get_html(url):
    return requests.get(url).text


soup = BeautifulSoup(get_html(url), 'html.parser')
a = soup.find_all('a')
dict_corp = []
for k in a:
    if "name" in str(k):
        dict_corp.append({"title": k.text, "pages": []})

dict_check = 0

for i in range(len(a)):
    if a[i].text == dict_corp[dict_check]["title"]:
        if dict_check < len(dict_corp) - 1:
            dict_check += 1
    else:
        dict_corp[dict_check - 1]["pages"].append(a[i].get('href'))

textes = ""
for k in dict_corp:
    ct = 0

    for p in k["pages"]:
        ct += 1
        #print(p)
        try:
            html_page = get_html(p)
        except:
            print(f"Skiping {p}")
            continue

        sp = BeautifulSoup(html_page, 'html.parser')
        #Récuperer les elements "Div" avec comme classe A, B, C ...
        paragraphs = sp.find_all("p")
        # Remove the <a> tags from all the paragraphs
        for p in paragraphs:
            for a in p.find_all('a'):
                a.decompose()

        text = "\n".join(p.text.strip() for p in paragraphs)
        textes += text + "\n\n\n"

f = open(fichier_destination, "w", encoding="utf-8")
f.write(textes)
f.close()
