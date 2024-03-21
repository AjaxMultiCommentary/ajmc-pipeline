import requests
import os


def get_zotero_data(zotero_id):
    resp = requests.get(
        f'{os.getenv("ZOTERO_API_URL", "https://api.zotero.org/groups/GROUP")}/items/{zotero_id}',
        headers={"Authorization": f"Bearer {os.getenv('ZOTERO_API_TOKEN', '')}"},
    )
    return resp.json()["data"]
