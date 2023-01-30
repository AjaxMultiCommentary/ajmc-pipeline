from ajmc.nlp.text_scraping.utils import scrape_resumptiontoken_oai


# CHECKED 2023-01-24

def scrape_openedition_oai():
    scrape_resumptiontoken_oai(base_request='http://oai.openedition.org/?verb=ListRecords',
                               additional_request_term='&metadataPrefix=qdc',
                               output_dir='/scratch/sven/openeditions_metadata/qdc_xml',
                               resume=True)


if __name__ == '__main__':
    scrape_openedition_oai()




