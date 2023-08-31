from ajmc.ajmc_corpora.scraping_utils import scrape_resumptiontoken_oai


def scrape_openedition_oai():
    scrape_resumptiontoken_oai(base_request='http://oai.openedition.org/?verb=ListRecords',
                               additional_request_term='&metadataPrefix=qdc',
                               output_dir='/scratch/sven/openeditions_metadata/qdc_xml',
                               resume=True)


if __name__ == '__main__':
    scrape_openedition_oai()
