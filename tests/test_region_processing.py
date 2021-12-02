import oclr.utils.region_processing as rp
from text_importer.pagexml import PagexmlPage
from commons.utils import timer
from oclr.utils.geometry import Shape

page_id = 'sophoclesplaysa05campgoog_0146'
page = PagexmlPage(page_id)
via_project = page._get_parent("commentary").via_project

regions = rp.get_page_regions(page, via_project)



def test_order_olr_regions():
    assert False
