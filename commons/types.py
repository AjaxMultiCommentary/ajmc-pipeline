"""This file contains the custom types used in ajmc"""
from typing import Union, List, Tuple

CommentaryType = Union['PagexmlCommentary']
ElementType = Union['PageType', 'PagexmlElement', 'HocrElement', 'Region']
RectangleType = List[Tuple[int, int]]
PageType = Union['PagexmlPage', 'HocrPage']
PathType = Union[str, bytes]
