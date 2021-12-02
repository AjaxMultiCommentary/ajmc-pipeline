⚠️ This is not up-to-date ⚠️

# A few words on the code

As mentionned above, `annotation_helper` performs the following tasks :

1. Converting Lace-annotations (.svg) in VIA2 annotations (.csv)
2. Detecting zones in image-files
3. Adding detected zones to Lace-annotations. 


**Converting Lace-annotations (.svg) in VIA2 annotations (.csv)** is done in `svg_converter.py`. The idea is to
transform svg-annotation into VIA2 annotations by converting the vectorial coordinates of rectangles 
into pixel-coordinates. 

**Detecting zones in image-files** is done using `cv2.dilation`. This dilates recognized letters-contours to recognize 
wider structures. The retrieved rectangles are then shrinked back to their original size. This can be seen
when drawing rectangles on images. 

**Adding detected zones to Lace-annotations** is done by comparing detected rectangles to lace-rectangles. If a 
detected rectangle does not match any Lace-rectangles, it is added to the final list. 

