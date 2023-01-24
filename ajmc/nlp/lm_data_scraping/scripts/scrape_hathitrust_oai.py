from ajmc.nlp.lm_data_scraping.utils import scrape_resumptiontoken_oai


# To be checked before run (if applicable)
scrape_resumptiontoken_oai(base_request='https://quod.lib.umich.edu/cgi/o/oai/oai?verb=ListRecords',
                           additional_request_term='&metadataPrefix=oai_dc&set=hathitrust',
                           output_dir='/scratch/sven/hathitrust_metadata/oai_dc_xml',)