import random
import time

from helium import *


auth_info = {'Username': "sophiachri", 'PSWD': "lemotdepasse"}
wait_time_range = [4, 8]


def getDict(URL: str) -> dict:
    '''
        Input:
        - URL The url of the dictionnary on the TLG site

        Output:
            A dictionnary(In python) with in index the word and in value the definition as a string
    '''

    #Verifications de l'URL et de la page web
    if type(URL) != str:
        raise ValueError("Error: Format\n", "You have to give an URL as a string format")

    # try:
    #     reponse = requests.get(URL)
    # except:
    #     raise ValueError("You have to give an real URL")
    #
    # if not reponse.ok:
    #      raise ValueError("Error:", reponse.status_code)

    #Creation du dictionnaire de sortie
    outing_value = dict()

    browser = start_firefox(URL)
    browser.switch_to.alert.accept()

    wait_until(Button('Login').exists)
    write(auth_info["Username"], into=browser.find_element("id", "username"))
    write(auth_info["PSWD"], into=browser.find_element("id", "password"))
    click('Login')
    wait_until(Link('Logout').exists)

    nbr_entries = int(Text(to_right_of='Entries').value)
    # for i in range(1, nbr_entries + 1):
    for i in range(1, 100):
        entry_id = "defn_" + str(i)

        time.sleep(random.randrange(wait_time_range[0] * 100, wait_time_range[1] * 100) / 100);
        try:  # why ⚠️
            click(CheckBox("I am using this site in accordance with its Terms of Use."))
            time.sleep(random.randrange(wait_time_range[0] * 100, wait_time_range[1] * 100) / 100);
            click(browser.find_element("id", "submit_select_srch"))
            time.sleep(random.randrange(wait_time_range[0] * 100, wait_time_range[1] * 100) / 100);
        except:
            print("Pas de captcha")
        print(entry_id)
        entry = browser.find_element("id", entry_id).text
        entry = entry.split(" ", 1)
        outing_value[entry[0]] = entry[1]

        click(browser.find_element("id", "defnDownButton"))

    kill_browser()
    return outing_value

# test = getDict('http://stephanus.tlg.uci.edu/lsj/')

# todo : diviser la boucle générale en sous boucle de ~250 iter et écrire le dico dans un json
# Todo : récuperer trois définitions d'un coup [blbal](https://.)
# todo Vérifier pour l'obtention des liens (si ce n'est pas trop compliqué).
