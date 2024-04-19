from pathlib import Path

# Read an XML file with bs4
from bs4 import BeautifulSoup

file_path = Path('/Users/sven/Desktop/lawiki-20240320-pages-articles-multistream.xml')
soup = BeautifulSoup(file_path.read_text('utf-8'), features='xml')

# find all the element named 'page' in the soup
pages = soup.find_all('page')

print(len(pages))

print(pages[0].prettify())

# We now get the text of the first page
text = pages[0].text

# We now estimate the size of the text in gb
size = len(text) / 1e9
