import requests
import os


def get_zotero_data(zotero_id):
    resp = requests.get(
        f'{os.getenv("ZOTERO_API_URL", "https://api.zotero.org/groups/GROUP")}/items/{zotero_id}',
        headers={"Authorization": f"Bearer {os.getenv('ZOTERO_API_TOKEN', '')}"},
    )
    data = resp.json()["data"]
    extra = data["extra"]
    extra_pairs = [line.split(": ") for line in extra.split("\n")]
    extra_dict = dict((k, v) for [k, v] in extra_pairs)

    data["extra"] = extra_dict

    return data
