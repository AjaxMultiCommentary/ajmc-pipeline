import argparse
import json
import random
import time
from pathlib import Path
from urllib.parse import urljoin

import helium as hl
from bs4 import BeautifulSoup

BASE_URL = 'https://logeion.uchicago.edu/'

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)


def get_word_from_dir(dir_: Path) -> str:
    return list((dir_ / 'words.txt').read_text(encoding='utf-8').split('\n'))[-2].strip()


def walk_logeion(word, stop, output_dir: Path, nav):
    """Walks the dict"""

    hl.click(nav.find_element_by_css_selector('li.word.ng-scope.selected-word').find_element_by_xpath('following-sibling::li[1]'))

    i = 0

    # Start the scraping
    while word != stop:

        i += 1
        if i % 200 == 0:
            print(f'----- Current word: {word} ------')

        time.sleep(random.randint(35, 43) / 100)
        items = nav.find_elements_by_tag_name('md-tab-item')

        if len(items) > 0:
            hl.click(items[0])

        soup = BeautifulSoup(hl.get_driver().page_source, 'html.parser')
        dicts_titles = soup.find_all('md-tab-item')

        # Get the short definition
        word = soup.find('h1', class_='ng-binding').text.strip()
        short_definition = soup.find('li', class_='ng-binding ng-scope')
        if short_definition is not None:
            (output_dir / 'short_definitions.jsonl').open('a+', encoding='utf-8').write(
                    json.dumps({word: short_definition.text}, ensure_ascii=False) + '\n')

        # Get the other definitions

        for i in range(len(items)):
            try:
                (output_dir / f'{dicts_titles[i].text}.jsonl').open('a+', encoding='utf-8').write(
                        json.dumps({word: soup.find('p', attrs={'ng-bind-html': 'content | html', 'class': 'ng-binding'}).text},
                                   ensure_ascii=False) + '\n')
            except IndexError:
                print('Warning on word: ', word)

            hl.press(hl.RIGHT)
            hl.press(hl.ENTER)
            soup = BeautifulSoup(hl.get_driver().page_source, 'html.parser')

        time.sleep(random.randint(11, 15) / 100)
        (output_dir / 'words.txt').open('a+', encoding='utf-8').write(word + '\n')
        hl.click(nav.find_element_by_css_selector('li.word.ng-scope.selected-word').find_element_by_xpath('following-sibling::li[1]'))

    print(f'**************** Scraping completed, finished with: {word} ****************')


def auto_restarting_walk(output_dir):
    """Restarts the scraping from the last word in the words.txt file"""

    # Get the start word and the stop word
    word = get_word_from_dir(output_dir)
    stop = json.loads((output_dir / 'metadata.json').read_text(encoding='utf-8'))['stop']
    start_url = urljoin(BASE_URL, word)
    print(f'************* Starting from word: {word} *************')

    nav = hl.start_firefox(start_url, headless=True)
    nav.implicitly_wait(5)

    try:
        walk_logeion(word, stop, output_dir, nav)
        nav.quit()

    except KeyboardInterrupt:
        nav.quit()
        return

    except Exception as e:
        print(e)
        nav.quit()
        auto_restarting_walk(output_dir)


if __name__ == '__main__':
    output_dir = Path(parser.parse_args().output_dir)
    auto_restarting_walk(output_dir)
