from text_importer.base_classes import Region
from text_importer import pagexml

page_id = 'sophoclesplaysa05campgoog_0146'
page = pagexml.PagexmlPage(page_id)


def test_page():
    assert type(page.regions) == list
    assert type(page.regions[0]) == type(Region)
