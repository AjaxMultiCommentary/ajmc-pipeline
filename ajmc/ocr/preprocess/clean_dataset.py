import os
import sys
import json
# import spacy
import shutil
# import chardet
# import fasttext
import numpy as np
from tqdm import tqdm
# from lingua import Language, LanguageDetectorBuilder
# from spacy.language import Language
# from spacy_langdetect import LanguageDetector
from googletrans import Translator, LANGUAGES
# from langdetect import detect_langs

sys.path.append("../")
from run.run_tesseract import PROJECT_DIR, PARENT_DIR, RAW_COMMENTARY_DIRS, TRAIN_COMMENTARY_DIRS, TESSDATA_DIR, TESSDATA_BEST_DIR, POGRETRA_COMMENTARY_DIRS
from run.run_tesseract import get_commentary_dir

# class LanguageIdentification:

#     def __init__(self):
#         pretrained_lang_model = "code/lid.176.bin"
#         self.model = fasttext.load_model(pretrained_lang_model)

#     def predict_lang(self, text):
#         predictions = self.model.predict(text, k=3) # returns top 2 matching languages
#         return predictions

# sentence_detection = False

# def get_lang_detector(nlp, name):
#     return LanguageDetector()

def count_freq(charset, text):
    for char in text:
        if char not in charset:
            charset[char] = 0
        charset[char] += 1

def clean_dataset(folder, target_lang=["greek"], img_suffix="png"):
    # for lang in target_lang:
    #     if lang.lower() not in ["greek"]:
    #         raise NotImplementedError(f"currently no support for language: {lang}")

    # spacy
    # nlp = spacy.load("el_core_news_sm") # 13M
    # nlp = spacy.load("el_core_news_md") # 41M
    # nlp = spacy.load("el_core_news_lg") # 543M
    # nlp = spacy.load("en_core_web_lg")
    # Language.factory("language_detector", func=get_lang_detector)
    # nlp.add_pipe('language_detector', last=True)
    
    # lingua
    # languages = [Language.GREEK, Language.ENGLISH, Language.LATIN]
    # detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # googletrans
    translator = Translator(raise_exception=True)

    # fasttext
    # LANGUAGE = LanguageIdentification()


    folder_basename = os.path.basename(folder)
    folder_parent = os.path.dirname(folder)
    clean_folder = os.path.join(folder_parent, f"{folder_basename}_cleaned")
    os.makedirs(clean_folder, exist_ok=False)
    log = {}
    raw_count = 0
    total_count = 0
    match_count = 0
    ignored = []
    charset_match = {}
    charset_ignored = {}
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

                # spacy
                # doc = nlp(gt_text)
                # lang_info = doc._.language
                # lang = lang_info["language"]

                # lingua
                # lang_info = detector.compute_language_confidence_values(gt_text)
                # print(lang_info)
                # lang = lang_info[np.argmax([item[1] for item in lang_info])][0].name if lang_info else "Not determined"

                # googletrans
                # lang_info = translator.detect_legacy(gt_text)
                lang_info = translator.detect(gt_text)
                lang_info = {"lang": lang_info.lang, "conf": lang_info.confidence}
                if isinstance(lang_info["lang"], list):
                    lang_info["lang"] = [LANGUAGES[item.lower()] for item in lang_info["lang"]]
                    lang = lang_info["lang"][np.argmax(lang_info["conf"])]
                else:
                    lang_info["lang"] = LANGUAGES[lang_info["lang"].lower()]
                    lang = lang_info["lang"]

                # langdetect
                # lang_info = detect_langs(gt_text)
                # print(lang_info)
                # lang = ""

                # fasttext
                # lang_info = LANGUAGE.predict_lang(gt_text)
                # lang = lang_info[0][np.argmax(lang_info[1])].replace("__label__", "")
                # print(lang_info)
                # lang_info = [item.tolist() if isinstance(item, np.ndarray) else item for item in lang_info]

                # document level language detection. Think of it like average language of the document!
                # print(lang)
                # if sentence_detection:
                #     # sentence level language detection
                #     for sent in doc.sents:
                #         print(sent, sent._.language)

                if lang.lower() not in target_lang:
                    ignored.append({"filename": gt_file, "text": gt_text, "detected language": lang_info})
                    count_freq(charset_ignored, gt_text)
                    continue

                match_count += 1
                count_freq(charset_match, gt_text)
                shutil.copyfile(gt_file, os.path.join(clean_folder, filename))
                shutil.copyfile(img_file, os.path.join(clean_folder, img_filename))
    except KeyboardInterrupt as e:
        print("Code is interrupted by user.")
    finally:
        log["target language"] = target_lang
        log["raw file counts"] = raw_count
        log["total img count"] = total_count
        log["match img count"] = match_count
        log["removed imgs"] = ignored
        log["charset for matched imgs"] = charset_match
        log["charset for removed imgs"] = charset_ignored
        output_file = os.path.join(clean_folder, "stats.json")
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(log, f_out, indent=2, ensure_ascii=False)

for subdir_name in POGRETRA_COMMENTARY_DIRS:
    subdir = get_commentary_dir(subdir_name, "pogretra", cleaned=False, create_if_missing=False)
    print(f"Now processing {subdir}")
    clean_dataset(subdir, target_lang=["greek"], img_suffix=".png")