"""
``ajmc.commons`` contains all the utilities (helpers, functions, objects, hard-coded variables) which are common (i.e. which must be accessible) to
task-specific repositories. These include notably:

* ``file_management`` utilities, which allow to handle files systematically in the ajmc's data organisation \
(see ``notebooks/data_organisation.ipnb`` for more) and to retrieve information from the various project spreadsheets.
* ``arithmetic.py`` contains helper maths function, mainly to deal with intervals, which are a common object in our Canonical format (of which more below).
* ``docstrings.py`` centralizes common function and class docstrings in a single place and provides a decorator to retrieve them easily.
* ``geometry.py`` provides helper functions and an object, ``Shape``, to deal with geometrical objects such as contours and bounding boxes.
* ``image.py`` provides helper functions and an object, ``AjmcImage``, to deal with images.
* ``miscellaneous.py`` receives everything which doesn't fit anywhere else. It notably contains generic functions and decorator, lazy objects for efficiency etc...
* ``variables`` contains all the hard-coded variables such as PATHS, COLORS, SPREADSHEET_IDS, CHARSETS and many more.
"""
