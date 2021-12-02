"""This module contains function to enhance via project jsons by reshaping regions and reorganising reading order."""

import json
from commons.variables import PATHS
import os
import cv2
from typing import List
from oclr.utils.geometry import Shape, is_rectangle_within_rectangle, remove_artifacts_from_contours
from oclr.utils import image_processing


with open('/Users/sven/ajmc/data/olr/via_project.json', "r") as f:
    via_project = json.loads(f.read())

pages = via_project["_via_img_metadata"]

page = pages['sophokle1v3soph_0000.png69464']
commentary_id = "sophokle1v3soph"
img_path = os.path.join(PATHS["base_dir"],commentary_id, PATHS["png"], page["filename"])

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
cv2.imwrite("/Users/sven/ajmc/data/olr/test_draw_.jpg", image)