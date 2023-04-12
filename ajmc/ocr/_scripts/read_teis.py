from pathlib import Path

from lxml import etree


tei_path = Path('/mnt/ajmcdata1/data/perseus/Classics/Sophocles/opensource/jebb.soph.aj_eng.xml')
tree = etree.parse(str(tei_path), parser=etree.XMLParser(no_network=False))

# get the root of the tree
root = tree.getroot()

#%%
a = ' '.join(root.itertext())
