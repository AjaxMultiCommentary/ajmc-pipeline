import text_importer.file_management as fm

page_id = "sophoclesplaysa05campgoog_0025"
commentary_id = page_id.split("_")[0]


def test_get_page_ocr_path():
    # Assert there is only a single ocr-file corresponding to page_id
    assert len(fm.get_page_ocr_path(page_id, "pagexml")) == 1


fm.get_page_ocr_path(page_id, commentary_id, "pagexml")