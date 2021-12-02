from typing import List
from commons.types import PageType


def get_page_region_dicts_from_via(page_id: str, via_project: dict) -> List[dict]:
    """Extract region-dicts of a page from`via_project`."""
    regions = []
    for key in via_project["_via_img_metadata"].keys():
        if page_id in key:
            regions = via_project["_via_img_metadata"][key]["regions"]
            break

    return regions


def select_page_regions_by_types(page: PageType,
                                 region_types: List[str]) -> List['Region']:
    return [r for r in page.regions if r.region_type in region_types]


def order_olr_regions(regions: List['Region']):
    """Orders elements according to reading order.

    This is a very simple algorithm to sort OLR or OCR elements according to reading order. This is particularly
    usefull for via regions, which are unordered in the via_dict. The algorithm works as follows:

        1. Look for the highest region.
        2. In case other regions start higher than the bottom of th highest region, take the leftest of them.

    Returns:
        List[Region]: The ordered list of regions.
    """

    ordered = []

    # Sort regions from highest to lowest (done only once, to speed up computation)
    regions.sort(key=lambda x: x.coords.xywh[1])

    # Select the top region
    while len(ordered) < len(regions):
        rest = [r for r in regions if r not in ordered]

        # see if there are other regions overlapping on the y-axis (this will include rest[0] itself).
        y_overlaps = [r for r in rest if r.coords.bounding_rectangle[0][1] < rest[0].coords.bounding_rectangle[2][1]]

        # select the lefter region
        ordered.append(sorted(y_overlaps, key=lambda x: x.coords.xywh[0])[0])

    return ordered
