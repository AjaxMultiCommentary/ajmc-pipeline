from ajmc.corpora import variables as vs
from ajmc.corpora.scraping_utils import scrape_resumptiontoken_oai

scrape_resumptiontoken_oai(base_request='https://archiv.ub.uni-heidelberg.de/propylaeumdok/cgi/oai2?verb=ListRecords',
                           additional_request_term='&metadataPrefix=oai_dc',
                           output_dir=(vs.ROOT_SCRAPE_DIR / 'propylaeum/metadata/oai_files'))
