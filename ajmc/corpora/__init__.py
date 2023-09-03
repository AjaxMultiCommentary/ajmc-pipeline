"""
``ajmc.corpora`` contains utils for scraping and handling corpora for the AjMC project.

Architecture
============

The functionalities of this package can be divided in three main categories:

* Scraping corpora from the web
* Cleaning and preparing downloaded corpora for further processing
* Processing and manipulating corpora

Scraping and cleaning
*********************

As each corpus has its own peculiarities, it also has its own scraping and cleaning script (see ``corpora._scripts``).

Processing
**********

``corpora`` provides a set of functions and objects for processing and manipulating corpora. Basically, each corpus has a type specified in its \
``metadata.json`` file. This type is used to determine which functions and objects to use for processing the corpus.

The main object is the ``Corpus`` object, which is a wrapper around the different corpus styles. See ``corpora_classes.py`` for more.

Basic usage
===========

.. code-block:: python
   from ajmc_corpora.corpora_classes import Corpus, PlainTextCorpus, TeiCorpus

   # Using `Corpus.auto_init` will automatically determine the type of the corpus
   corpus = Corpus.auto_init('remacle')  # returns a `PlainTextCorpus` object
   corpus = Corpus.auto_init('persee')  # returns a `TeiCorpus` object

   # You can also use a custom corpus class
   corpus = PlainTextCorpus('remacle')

    # Then a few WIP basic operation are available
   corpus.get_plain_text()  # returns a string containing the full plain text of the corpus
   corpus.write_plain_text()  # writes the plain text to a file

   corpus.get_regions()  # Gets the paragraphs of the corpus
   corpus.get_lines()  # Gets the lines of the corpus

   corpus.get_chunks(chunk_size=512, unit='character')  # Chunks the corpus into chunks of n words or chars, returns a list of strings.

"""
