from ajmc.commons.file_management import walk_dirs
from ajmc.corpora import variables as vs
from ajmc.corpora.corpora_classes import Corpus


DONE = [
    'forum_romanum',
    'corpus_scriptorum_latinorum',
    'canonical-latinLit',
    'canonical-greekLit',
    'perseus_secondary',
    'perseus_legacy',
    'First1KGreek',
    'propylaeum_BOOKS',
    'propylaeum_DOK',
    'agoraclass',
]

corpora_stats = {}

for corpus_id in walk_dirs(vs.ROOT_STORING_DIR):
    corpus_id = corpus_id.stem
    corpus_id = 'EpibauCorpus'
    if corpus_id in DONE:
        continue
    print('---------------------------------')
    print(corpus_id)
    try:
        corpus = Corpus.auto_init(corpus_id)
        corpora_stats[corpus_id] = len(corpus.get_plain_text())
        print(corpora_stats[corpus_id])
    except Exception as e:
        print('Skipping corpus:', corpus_id, e)
    break
