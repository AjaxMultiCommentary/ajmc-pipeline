import ajmc.text_processing.canonical_classes as cc
import ajmc.commons.variables as vars
import json
import os

WIKIDATA_HUCIT_MAPPINGS = {}

with open(os.path.dirname(__file__) + "/wikidata_hucit_mappings.json") as f:
    WIKIDATA_HUCIT_MAPPINGS = json.load(f)


def contains_primary_full(entities):
    return any([is_primary_full(entity) for entity in entities])


def is_primary_full(entity):
    return entity.label == "primary-full"


def is_entity_in_range(entity, word_range):
    return (
        word_range[0] <= entity.word_range[0]
        and (word_range[1] + 1) >= entity.word_range[1]
    )


def remove_trailing_character(s: str, c: str) -> str:
    if s.endswith(c):
        return s[0:-1]

    return s


def transform_f(s: str) -> str:
    if s.endswith("f."):
        without_f = s.replace("f.", "")

        try:
            n = int(without_f)
            return f"{n}-{n+1}"
        except ValueError:
            # Return the scope without "f." because we're using it in a CTS URN
            return without_f

    return s


class PrimaryFullEntity:
    def __init__(
        self,
        cts_urn: str,
        scopes: list[cc.CanonicalEntity],
        words: list[cc.CanonicalWord],
    ):
        self.cts_urn = cts_urn
        self.scopes = scopes
        self.words = words
        self.word_range = [words[0].word_range[0], words[-1].word_range[1]]
        self.url = self.to_url()

    def to_url(self):
        if len(self.scopes) == 0:
            return f"https://scaife.perseus.org/reader/{self.cts_urn}"
        else:
            return f"https://scaife.perseus.org/reader/{self.cts_urn}{self.resolve_scopes()}"

    def resolve_scopes(self):
        scope_first = self.scopes[0]
        scope_last = self.scopes[-1]
        scope_words = [
            w
            for w in self.words
            if w.word_range[0]
            in range(scope_first.word_range[0], scope_last.word_range[1] + 1)
        ]
        s = (
            "".join([w.text for w in scope_words])
            .replace("(", "")
            .replace(")", "")
            .replace(";", "")
            .replace(":", "")
            .replace("ff.", "")
        )
        s = transform_f(s)
        s = remove_trailing_character(s, ",")
        s = remove_trailing_character(s, ".")

        if len(s) > 0:
            return f":{s}"

        return ""


def make_primary_full_entities(commentary: cc.CanonicalCommentary):
    all_entities = commentary.children.entities
    primary_fulls = []

    for entity in all_entities:
        if entity.label == "primary-full":
            related_entities = [
                e for e in all_entities if is_entity_in_range(e, entity.word_range)
            ]
            primlits = [e for e in related_entities if e.label == "work.primlit"]
            scopes = [e for e in related_entities if e.label == "scope"]

            wikidata_id = next((e.wikidata_id for e in primlits), None)

            if wikidata_id is None:
                wikidata_id = next((e.wikidata_id for e in scopes), None)

                if wikidata_id is None:
                    continue

            cts_urn = WIKIDATA_HUCIT_MAPPINGS.get(wikidata_id, {}).get("cts_urn")

            if cts_urn is None or cts_urn == "":
                continue

            entity_words = commentary.children.words[
                entity.word_range[0] : (entity.word_range[1] + 1)
            ]

            if len(entity_words) == 0:
                continue

            primary_fulls.append(PrimaryFullEntity(cts_urn, scopes, entity_words))

    return primary_fulls


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
        self.primary_full_entities = make_primary_full_entities(self.commentary)

    def facsimile(self, page):
        return f"{self.ajmc_id}/{page.id}"

    def frontmatter(self):
        raise NotImplementedError(
            "frontmatter() needs to be implemented in the child class"
        )

    def title(self):
        return self.bibliographic_data["title"]
