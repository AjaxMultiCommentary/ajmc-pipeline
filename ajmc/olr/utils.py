from typing import List, Optional, Dict

from ajmc.commons import variables
from ajmc.commons.arithmetic import compute_interval_overlap
from ajmc.commons.miscellaneous import get_olr_splits_page_ids


def get_page_region_dicts_from_via(page_id: str, via_project: dict) -> List[dict]:
    """Extract region-dicts of a page from`via_project`."""
    regions = []
    for key in via_project["_via_img_metadata"].keys():
        if page_id in key:
            regions = via_project["_via_img_metadata"][key]["regions"]
            break

    return regions


def select_page_regions_by_types(page: 'OcrPage',
                                 region_types: List[str]) -> List['OlrRegion']:
    return [r for r in page.children['region'] if r.region_type in region_types]


def sort_to_reading_order(elements: list,
                          overlap_thresh: float = 0.6):
    """Orders elements according to reading order.

    This is a very simple algorithm to sort OLR or OCR elements according to reading order. This is particularly
    usefull for via regions, which are unordered in the via_dict. The algorithm works as follows:

        1. Order the elements from highest to lowest.
        2. Take the highest element.
        3. In case other elements have significant y-overlap with the highest element, take the leftest of them.
        4. Iterate until all elements are reordered.

    Note:
        This will NOT order column-separated lines correctly !

    Returns:
        list: The list of ordered elements.
    """

    ordered = []

    # Sort regions from highest to lowest (done only once, to speed up computation)
    elements.sort(key=lambda x: x.bbox.xywh[1])

    # Select the top region
    while len(ordered) < len(elements):
        rest = [e for e in elements if e not in ordered]

        # y_overlaps = [r for r in rest if r.bbox.bbox[0][1] < rest[0].bbox.bbox[1][1]]
        # see if there are other regions overlapping on the y-axis (this will include rest[0] itself).

        overlapping_candidates = []

        for r in rest:  # for each remainding
            # Compute the y-overlap it this element has with highest element (rest[0])
            y_overlap = compute_interval_overlap(i1=(rest[0].bbox.bbox[0][1],
                                                     rest[0].bbox.bbox[1][1]),
                                                 i2=(r.bbox.bbox[0][1],
                                                     r.bbox.bbox[1][1]))
            # If the y-overlap are above `overlap_threshold`, append the element to the `overlapping_candidates`
            if y_overlap > overlap_thresh * rest[0].bbox.height \
                    or y_overlap > overlap_thresh * r.bbox.height:
                overlapping_candidates.append(r)

        ordered.append(sorted(overlapping_candidates, key=lambda x: x.bbox.xywh[0])[0])  # select the leftest element

    return ordered


def get_olr_region_counts(commentaries: List['CanonicalCommentary'],
                          splits: Optional[List[str]] = None,
                          fine_to_coarse: Optional[Dict[str, str]] = None) -> dict:
    """Get olr regions counts from commentary.

     Args:
        commentaries: A canonical commentary object
        splits: The desired splits, eg `['train', 'test']`.
        fine_to_coarse: A mapping from fine regions to coarse.
    """
    # Initialize the counts
    region_types_counts = {(fine_to_coarse[rt] if fine_to_coarse else rt): 0 for rt in variables.ROIS+['pages', 'total']}


    for commentary in commentaries:
        # ⚠️ Get the list of groundtruth pages ONLY (remember that `commentary` zones are annotated on all pages !)
        gt_pages_ids = get_olr_splits_page_ids(commentary.id, splits)
        gt_pages = [p for p in commentary.children['page'] if p.id in gt_pages_ids]

        # Do the counts

        for p in gt_pages:
            region_types_counts['pages'] += 1
            for r in p.children['region']:
                if r.info['region_type'] != 'line_region':
                    region_types_counts['total'] += 1
                    region_types_counts[fine_to_coarse[r.info['region_type']] if fine_to_coarse else r.info['region_type']] += 1



    return region_types_counts