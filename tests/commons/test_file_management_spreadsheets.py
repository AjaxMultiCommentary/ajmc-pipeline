import ajmc.commons.file_management
import tests.sample_objects as so

# Prepare test pages
valid_full_gt_pages = []  # This contains the pages which are entirely annotated
valid_comm_gt_pages = []  # This contains the pages where only commentary sections are annotated


for v in so.sample_ocrcommentary.via_project['_via_img_metadata'].values():

    if any([r['region_attributes']['text'] not in ['commentary', 'undefined'] for r in v['regions']]):
        valid_full_gt_pages.append(v['filename'].split('.')[0])

    elif all([r['region_attributes']['text'] in ['commentary', 'undefined'] for r in v['regions']]) and \
            any([r['region_attributes']['text'] in ['commentary'] for r in v['regions']]):
        valid_comm_gt_pages.append(v['filename'].split('.')[0])

invalid_full_gt_pages = valid_full_gt_pages[:-1] + ['added_page_0001']


def test_check_via_spreadsheet_conformity():
    assert ajmc.commons.file_management.utils.check_via_spreadsheet_conformity(so.sample_ocrcommentary.via_project,
                                                                               valid_full_gt_pages) == (set(), set())
    assert ajmc.commons.file_management.utils.check_via_spreadsheet_conformity(so.sample_ocrcommentary.via_project,
                                                                               valid_comm_gt_pages,
                                                                               check_comm_only=True) == (set(), set())
    assert [len(x) for x in
            ajmc.commons.file_management.utils.check_via_spreadsheet_conformity(so.sample_ocrcommentary.via_project,
                                                                                invalid_full_gt_pages)] == [1, 1]
