from typing import List

from ajmc.commons.arithmetic import compute_interval_overlap


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
    return [r for r in page.regions if r.region_type in region_types]


def sort_to_reading_order(elements: list,
                          # Todo : this can only be a list of elements with coords. Do a mother class for text elements ?
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
    elements.sort(key=lambda x: x.coords.xywh[1])

    # Select the top region
    while len(ordered) < len(elements):
        rest = [e for e in elements if e not in ordered]

        # y_overlaps = [r for r in rest if r.coords.bounding_rectangle[0][1] < rest[0].coords.bounding_rectangle[2][1]]
        # see if there are other regions overlapping on the y-axis (this will include rest[0] itself).

        overlapping_candidates = []

        for r in rest:  # for each remainding
            # Compute the y-overlap it this element has with highest element (rest[0])
            y_overlap = compute_interval_overlap(i1=(rest[0].coords.bounding_rectangle[0][1],
                                                     rest[0].coords.bounding_rectangle[2][1]),
                                                 i2=(r.coords.bounding_rectangle[0][1],
                                                     r.coords.bounding_rectangle[2][1]))
            # If the y-overlap are above `overlap_threshold`, append the element to the `overlapping_candidates`
            if y_overlap > overlap_thresh * rest[0].coords.height \
                    or y_overlap > overlap_thresh * r.coords.height:
                overlapping_candidates.append(r)

        ordered.append(sorted(overlapping_candidates, key=lambda x: x.coords.xywh[0])[0])  # select the leftest element

    return ordered
