"""This module contains function to enhance via project jsons by reshaping regions and reorganising reading order."""

import json
from common_utils.variables import PATHS
import os
import cv2
from typing import List
from common_utils.geometry import Shape
from common_utils import image_processing

with open('/data/olr/via_project.json', "r") as f:
    via_project = json.loads(f.read())

pages = via_project["_via_img_metadata"]

page = pages['sophokle1v3soph_0000.png69464']
commentary_id = "sophokle1v3soph"
img_path = os.path.join(PATHS["base_dir"], commentary_id, PATHS["png"], page["filename"])

image = cv2.imread(img_path)

contours: List[Shape] = image_processing.find_contours(image, do_binarize=True)

#%%

[p for p in contours[0].points[:,0]]
def is_intersecting(s1: Shape, s2:Shape):
    # todo performance
    intersects = False
    for p1 in s1.points:
        for p2 in s2.points:
            if None:
                pass
            
#%%
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cv2.drawContours(image, contours, i, color = 255, thickness=-1 )

cv2.imshow("coucou", image)
contours[0]
cv2.imwrite("/data/olr/test_draw_.jpg", image)


#%%

## TOKEEP : verifies via integrity (here only making sure that pages contains either 0 or 2 commentary sections.
import json
with open('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/lestragdiesdeso00tourgoog/olr/via_project.json', "r") as f:
    via_project = json.loads(f.read())


for key, page in via_project['_via_img_metadata'].items():
    regions_count = 0
    for region in page['regions']:
        if region['region_attributes']['text']=='commentary':
            regions_count+=1
    if regions_count != 0 and regions_count!=2:
        print(key)
