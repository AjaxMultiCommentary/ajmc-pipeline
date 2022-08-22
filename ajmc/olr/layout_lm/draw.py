"""Draw utilities, notably to draw word labels."""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple

def draw_page_labels(img: 'PIL.Image',
                     words: List['OcrWord'],
                     labels: List[str],
                     labels_to_colors,
                      ):
    """Highlights words bounding boxes with the color corresponding to their label."""

    # Create image and draw
    draw_img = img.convert('RGB')
    draw = ImageDraw.Draw(draw_img, 'RGBA')

    for word, label in zip(words, labels):
        xy = tuple([word.bbox.bbox[0], word.bbox.bbox[2]])
        draw.rounded_rectangle(xy=xy, fill=labels_to_colors[label], radius=4)

    return draw_img

# Todo : Why the heck should this be a function ? This IS a funking script
def draw_caption(img: 'PIL.Image',
                 labels_to_colors: Dict[str, Tuple[int,int,int]]):

    draw_img = img.convert('RGB')

    # Draw captions
    base_img_width, base_img_height = img.size

    margin_top = int(base_img_height / 10)
    margin_left = int(base_img_width / 10)

    font = ImageFont.truetype("CourierNewTTF.ttf", size=int(base_img_width / 35))
    title_font = ImageFont.truetype("ORATOR10.ttf", size=int(base_img_width / 25))

    _, font_height = font.getsize('a')
    pad = int(font_height / 6)
    line_space = int(1.5 * font_height)

    # Create an extra margin for caption
    caption_width = max([font.getsize(label)[0] for label in labels_to_colors.keys()]) + 2 * margin_left
    new_img_width = caption_width + base_img_width
    new_draw_img = Image.new('RGB', (new_img_width, base_img_height), color=(255, 255, 255, 255))
    new_draw_img.paste(img, box=(caption_width, 0))
    draw = ImageDraw.Draw(new_draw_img, 'RGBA')

    # Create caption title
    title = "LABELS"
    draw.text(xy=(margin_left, margin_top), text=title, font=title_font, fill=(0, 0, 0))
    _, title_height = title_font.getsize(title)

    previous_y = margin_top + (2 * title_height)

    for i, (label, color) in enumerate(labels_to_colors.items()):
        xy = (margin_left, previous_y)
        w, h = font.getsize(label)
        draw.rounded_rectangle(xy=((xy[0] - pad, xy[1] - pad), (xy[0] + w + pad, xy[1] + h + pad)),
                               fill=color, radius=int(font_height / 8))

        draw.text(xy=xy, text=label, font=font, fill=(0, 0, 0))
        previous_y += line_space + font_height

    # Draw separation line
    draw.line(xy=[(caption_width, margin_top), (caption_width, base_img_height - margin_top)], fill=(0, 0, 0), width=3)

    return new_draw_img