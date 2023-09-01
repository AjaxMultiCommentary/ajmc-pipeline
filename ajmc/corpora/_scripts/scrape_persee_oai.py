from pathlib import Path

from ajmc.corpora.scraping_utils import scrape_resumptiontoken_oai

PERSEE_AJMC_COLL_IDS = ['crai', 'jds', 'mesav', 'piot', 'reg', 'vita', 'bude', 'camed', 'scrip', 'ccgg', 'nauti',
                        'rht', 'bec', 'bch', 'bch', 'mefr', 'efr', 'metis', 'gaia', 'cehm', 'thlou', 'dha', 'anata',
                        'antiq', 'rscir', 'medi', 'rea', 'anami', 'topoi', 'mcm', 'keryl',
                        'pouil', 'piot', 'minf', 'mesav', 'rem', 'rhfdf', 'anatv', 'anata',
                        'anatm', 'ista', 'dha', 'girea', 'mom']

output_dir = Path('/mnt/ajmcdata1/data/persee/data')
output_dir.mkdir(parents=True, exist_ok=True)

for coll in PERSEE_AJMC_COLL_IDS:
    base_request = f'http://oai.persee.fr/oai?verb=ListRecords'
    additional_request_term = f'&metadataPrefix=tei&set=persee:serie-{coll}:doc'

    scrape_resumptiontoken_oai(base_request=base_request,
                               output_dir=output_dir,
                               additional_request_term=additional_request_term,
                               resume=False,
                               filename_prefix=f'data_{coll}')
