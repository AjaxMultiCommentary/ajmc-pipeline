import ajmc.text_processing.canonical_classes as cc
import ajmc.commons.variables as vars
import os


class ExportableCommentary:
    def __init__(self, ajmc_id, bibliographic_data) -> None:
        canonical_path = vars.COMMS_DATA_DIR / ajmc_id / "canonical"
        filename = [
            f for f in os.listdir(canonical_path) if f.endswith("_tess_retrained.json")
        ][0]
        json_path = canonical_path / filename

        self.ajmc_id = ajmc_id
        self.bibliographic_data = bibliographic_data
        self.commentary = cc.CanonicalCommentary.from_json(json_path=json_path)
        self.filename = f"tei/{ajmc_id}.xml"
        self.tei = None

    def facsimile(self, page):
        return f"{self.ajmc_id}/{page.id}"

    def title(self):
        return self.bibliographic_data["title"]
