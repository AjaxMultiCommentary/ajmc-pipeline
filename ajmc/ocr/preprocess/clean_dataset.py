import os
import re
import sys
import json
from httpcore import ReadTimeout
import spacy
import shutil
import fasttext
import numpy as np
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from googletrans import Translator, LANGUAGES
from langdetect import detect_langs, lang_detect_exception

sys.stdout.reconfigure(encoding='utf-8') # useful for printing greek texts

PROJECT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(PROJECT_DIR))

from ajmc.ocr.run.run_tesseract import PARENT_DIR, RAW_COMMENTARY_DIRS, TRAIN_COMMENTARY_DIRS, TESSDATA_DIR, TESSDATA_BEST_DIR, POGRETRA_COMMENTARY_DIRS
from ajmc.ocr.run.run_tesseract import get_commentary_dir

def count_freq(charset, text):
    for char in text:
        if char not in charset:
            charset[char] = 0
        charset[char] += 1

class LangIdentifier:
    def __init__(self, method="googletrans", **kwargs):
        self.extra_config = kwargs
        # spacy
        if method.lower() in ["spacy"]:
            self.mode = "spacy"
            spacy_model = kwargs.get("spacy_model", "el_core_news_md")
            nlp = spacy.load(spacy_model)
            Language.factory("language_detector", func=lambda model, name: LanguageDetector())
            nlp.add_pipe('language_detector', last=True)
            self.model = nlp
        
        # lingua
        elif method.lower() in ["lingua"]:
            self.mode = "lingua"
            languages = kwargs.get("languages", [Language.GREEK, Language.ENGLISH, Language.LATIN])
            detector = LanguageDetectorBuilder.from_languages(*languages).build()
            self.model = detector

        # googletrans
        elif method.lower() in ["googletrans"]:
            self.mode = "googletrans"
            translator = Translator(raise_exception=True)
            self.model = translator

        # langdetect
        elif method.lower() in ["langdetect"]:
            self.mode = "langdetect"
            self.model = None

        # fasttext
        elif method.lower() in ["fasttext"]:
            self.mode = "fasttext"
            pretrained_lang_model = "code/lid.176.bin"
            self.model = fasttext.load_model(pretrained_lang_model)

        elif method.lower() in ["threshold"]:
            self.mode = "threshold"
            self.threshold = kwargs.get("threshold", 0.5)
            self.model = re.compile(r'([\u0373-\u03FF]|[\u1F00-\u1FFF]|\u0300|\u0301|\u0313|\u0314|\u0345|\u0342|\u0308|\s|\d)')
        
        else:
            raise NotImplementedError(f"no such method for language identification: {method}")

    def detect(self, gt_text):
        # spacy
        if self.mode.lower() in ["spacy"]:
            doc = self.model(gt_text)
            lang_info = doc._.language
            lang = lang_info["language"]
            conf = -1

        # lingua
        elif self.mode.lower() in ["lingua"]:
            lang_info = self.model.compute_language_confidence_values(gt_text)
            print(lang_info)
            lang = lang_info[np.argmax([item[1] for item in lang_info])][0].name if lang_info else "Not determined"
            conf = np.max([item[1] for item in lang_info]) if lang_info else -1

        # googletrans
        elif self.mode.lower() in ["googletrans"]:
            # lang_info = translator.detect_legacy(gt_text)
            while True:
                try:
                    lang_info = self.model.detect(gt_text)
                    break
                except ReadTimeout as e:
                    print(f"googletrans failed with ReadTimeout. Retrying...")
            lang_info = {"lang": lang_info.lang, "conf": lang_info.confidence}
            if isinstance(lang_info["lang"], list):
                lang_info["lang"] = [LANGUAGES[item.lower()] for item in lang_info["lang"]]
                lang = lang_info["lang"][np.argmax(lang_info["conf"])]
                conf = np.max(lang_info["conf"])
            else:
                lang_info["lang"] = LANGUAGES[lang_info["lang"].lower()]
                lang = lang_info["lang"]
                conf = lang_info["conf"]

        # langdetect
        elif self.mode.lower() in ["langdetect"]:
            try:
                lang_info = detect_langs(gt_text)
                lang = lang_info[0].lang
                conf = lang_info[0].prob

            except lang_detect_exception.lang_detect_exception.LangDetectException as e:
                print(f"langdetect failed with error: {e}")
                lang = "Not determined"
                conf = -1
                lang_info = {"lang": lang, "conf": conf}

        # fasttext
        elif self.mode.lower() in ["fasttext"]:
            lang_info = self.model.predict(gt_text, k=3)
            lang = lang_info[0][np.argmax(lang_info[1])].replace("__label__", "")
            conf = np.max(lang_info[1])
            print(lang_info)
            lang_info = [item.tolist() if isinstance(item, np.ndarray) else item for item in lang_info]

        # threshold
        elif self.mode.lower() in ["threshold"]:
            tmp = self.model.findall(gt_text)
            conf = len(tmp) / len(gt_text)
            lang = "greek" if conf >= self.threshold else "non-greek"
            lang_info = {"lang": lang, "conf": conf}

        else:
            raise NotImplementedError(f"Language identification model created, but it doesn't have an implementation for detection: {self.mode}")

        return lang_info, lang, conf
        

def clean_dataset(folder, clean_folder, target_lang=["greek"], img_suffix="png", method="googletrans", **kwargs):
    
    model = LangIdentifier(method, **kwargs)

    log = {}
    raw_count = 0
    total_count = 0
    match_count = 0
    ignored = []
    charset_match = {}
    charset_ignored = {}
    conf_stats = []
    try:
        for filename in tqdm(list(os.listdir(folder))):
            raw_count += 1
            if not filename.endswith(".gt.txt"):
                continue
            img_filename = filename.replace(".gt.txt", img_suffix)
            img_file = os.path.join(folder, img_filename)
            if not os.path.isfile(img_file):
                continue
            total_count += 1
            gt_file = os.path.join(folder, filename)
            with open(gt_file, "r", encoding="utf-8") as f_in:
                gt_text = f_in.read()

                # print(gt_text.encode("utf-8").decode("latin1"))

                # print(repr(gt_text).encode("utf-8").decode("latin1"))
                # input("continue?")

                lang_info, lang, conf = model.detect(gt_text)

                conf_stats.append(conf)

                if lang.lower() not in target_lang and LANGUAGES.get(lang.lower(), "unknown") not in target_lang:
                    count_freq(charset_ignored, gt_text)
                    ignored.append({"filename": gt_file, "text": gt_text, "detected language": lang_info})
                    continue

                match_count += 1
                count_freq(charset_match, gt_text)
                shutil.copyfile(gt_file, os.path.join(clean_folder, filename))
                shutil.copyfile(img_file, os.path.join(clean_folder, img_filename))

    except KeyboardInterrupt as e:
        print("Code is interrupted by user.")

    finally:
        log["target language"] = target_lang
        log["detector configs"] = {"method": method, "configs": model.extra_config}
        log["raw file counts"] = raw_count
        log["total img count"] = total_count
        log["match img count"] = match_count
        log["match img confidence stats (5%, 10%, 25%, 50%, 75%)"] = np.quantile(conf_stats, [.05, .1, .25, .50, .75]).tolist()
        log["removed imgs"] = ignored
        log["charset for matched imgs"] = charset_match
        log["charset for removed imgs"] = charset_ignored
        output_file = os.path.join(clean_folder, "stats.json")
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(log, f_out, indent=2, ensure_ascii=False)

method = "googletrans"
remove_existing_folders = False

for subdir_name in POGRETRA_COMMENTARY_DIRS:
    subdir = get_commentary_dir(subdir_name, "pogretra", cleaned_suffix="", create_if_missing=False)
    clean_folder = get_commentary_dir(subdir_name, "pogretra", cleaned_suffix=f"clean-{method}", create_if_missing=True)
    if remove_existing_folders:
        print(f"Now removing {clean_folder}")
        try:
            shutil.rmtree(clean_folder)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    else:
        print(f"Now processing {subdir}")
        if method.lower() in ["googletrans"]:
            clean_dataset(subdir, clean_folder, target_lang=["greek"], img_suffix=".png", method=method)
        elif method.lower() in ["threshold"]:
            clean_dataset(subdir, clean_folder, target_lang=["greek"], img_suffix=".png", method=method, threshold=0.5)