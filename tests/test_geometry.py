import numpy as np

import oclr.utils.geometry as geo
from commons.utils import timer
from text_importer.pagexml import PagexmlPage

points = [[1, 2], [3, 4], [5, 6], [1, 2]]
points_np = np.array(points)
shape = geo.Shape.from_points(points)
rect = np.array([[1, 1], [5, 7]])
point = np.array([3, 3])
page = PagexmlPage("sophoclesplaysa05campgoog_0146")

def test_bounding_rectangle():
    assert shape.bounding_rectangle.shape == (4, 2)


def test_from_array():
    assert geo.Shape.from_numpy_array(points_np).points == geo.Shape.from_points(points).points


#%%

points = [xy for w in page.words for xy in w.coords.bounding_rectangle]


#%%


#%%