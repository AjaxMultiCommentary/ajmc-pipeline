# PROCESSING FOR A SINGLE IMAGE
# Open the image as an AjmcImage
# Get the lines from the combined model
# Write the lines as pngs to a specific directory
# Lines will be named page_id_x_y_w_h.png # Todo here see if it's the best way

# Run the model on all the lines, using an ocriterdataset.
#  for each line, we generate a json file {'text': 'the text', 'word_bbox': [[x, y, w, h],...] }
