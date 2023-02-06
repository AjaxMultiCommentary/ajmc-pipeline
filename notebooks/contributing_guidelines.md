
# Contributing guidelines

This notebook provides contributors with a set of general principles, specific rules and best practices. Its goal to facilitate collaboration and maintenance of the library. It is highly recommended to read the `introduction_to_code.ipynb` before going further.


## General principles

I will first start by defining and illustrate the fundamental design principles of this codebase. All the examples below are voluntary simplified and do not reflect the actual codebase.

### The code should be systematic

#### Definition

Systematic means that similar situations, objects and pipelines should be handled similarly. It means that design principles should be shared across modules so that someone with a good knowledge of a module could easily predict how another module is written. It rimes with generality and consistency. It can conflict with the specificity of particular objects and with efficiency related issues. Hence :

> Be as systematic as you can, as specific as you must.

#### Examples

1. Two functions which are part of similar pipelines, share similar goals or deal with similar objects should have similar arguments.

✅ This is good practice:

```python
def compute_circle_area(radius: float) -> float:
    return numpy.pi * radius ** 2

def compute_circle_circumference(radius: float) -> float:
    return 2* numpy.pi * radius
```

❌ This should be avoided:

```python
def compute_circle_area(circle:'MyCircleObject') -> float:
    return numpy.pi * circle.radius ** 2

def compute_circle_circumference(radius: float) -> float:
    return 2 * numpy.pi * radius
```

In the second example, two functions doing very related operations unnecessarily require different types of arguments: the first expects a `MyCircleObject` while the second expects the circle's radius circle directly.

2. Closely related objects should behave similarly.

✅ This is good practice

```python
class Circle:
    #...
    @property
    def area(self):
        return numpy.pi * self.radius ** 2

class Rectangle:
    #...
    @property
    def area(self):
        return self.width * self.height
```

❌ This should be avoided

```python
class Circle:
    #...
    @property
    def area(self):
        return numpy.pi * self.radius ** 2

class Rectangle:
    #...
    def compute_area(self):
        return self.width * self.height
```

Here, there is no reason for calling a method in one case (`Rectangle.get_area()`) while retrieving the value of a property in the other (`Circle.area`). A similar example could have been to not implement `area` at all in `Rectangle`. Note that when an attribute is not relevant for an object, it is preferable to define and implement a null value than to potentially raise `AttributeError`s.

**In ajmc**. This is the case for words and lines. Though they are very similar objects, lines can have children, but words have none as they are the lowest level in our hierarchy. By convention, `Word.children.words` returns an empty list and does not raise an `AttributeError`. The same holds true for the parents of a commentary.

### The code should be centralized

#### Definition

The centrality is intimately connected to the [DRY principle](https://www.youtube.com/watch?v=IGH4-ZhfVDk). It states that every reused snippet of code should be *called* and not *duplicated*. Good use of this principle makes the maintenance of a codebase a lot easier. It can however conflict with the simplicity of imports as well as with the readability of the good. The number of usages, the simplicity of snippet as well the likelihood it will ever be changed or require maintenance should be good hints whether to centralise or not.


#### Examples

1. Docstrings.
**In ajmc**. This applies to code and to docstrings as well. Common docstrings should be written once in `commons.docstrings.docstrings` and then called with `docstring_formatter`:
```python
# docstrings is a simple dict
docstrings['circle_arg'] = 'A `MyCircleObject` allowing to call self.radius...'

@docstring_formatter(**docstrings)
def compute_circle_area(circle: 'MyCircleObject'):
    """Computes area.

    Args:
        circle: {circle_arg}
    """
```

2. Functions should expect minimal arguments.

Centrality is enhanced by the use of *minimal arguments*. To be able to reuse functions, have them expect only what they actually need to know in order to do what they do.

✅ This is good practice:

```python
def compute_text_len(text:str) -> int:
    return len(text)
```

❌ This should be avoided:

```python
def compute_line_len(line: 'OcrLine') -> int:
    return len(line.text)
```
Here, `compute_line_len` expects an `OcrLine` objets only to retrieve its text. The function is match to specialised and should actually be a method to the object, and not a free function in the loose. On the contrary, `compute_text_len` is much or general and can be easily called by several text containers.

More generally, pipeline-like functions should be avoided as much as possible. Something like :

```python
def run_outputs_evaluation_and_convert_to_table(outputs_file, some_output_related_parameter, output_path, table_styling):
    # Do something
    # ...
    return 
```
is terrible in terms of flexibility and centrality. It does allow you to run your evaluation (or whatever you are doing) without having your outputs as files and `some_output_related_parameter`. In future usages however, you might not have your outputs it the same format, maybe even not as files, and maybe you won't need `table_styling` anymore. Which is a perfect transition to the next point.

3. Functions should perform atomic actions

A single function should never perform two operations you think might be called independently. The example above is speaking for itself. Instead of `run_outputs_evaluation_and_convert_to_table`, have `run_outputs_evaluation` and `convert_evaluation_to_table`.

### The code should be efficient

#### Definition
As they constitute the backend of a dynamic platform, `ajmc`'s modules should be fluid and run smoothly. The code should require as little computation as possible.

#### Examples
1. Shared directory (i.e. `commons` and `text_processing` should require as little dependencies as possible. If you need heavy computation, it is certainly best to isolate your code in a task-specific folder which does not hamper the code base from running.

2. Object instantiation should be lazy.

`ajmc.commons.miscellaneous` provides two main utilities for lazyness: `lazy_property` and `LazyObject`. The former allows properties to be computed and then stored only when they are called.

✅ This is good practice:

```python
class AjmcImage:
    def __init__(self, path_to_file:str):
        self.path = path_to_file

    @lazy_property
    def matrix(self) -> np.ndarray:
        return cv2.imread(self.path)
```

❌ This should be avoided:
```python
class AjmcImage:
    def __init__(self, path_to_file:str):
        self.path = path_to_file
        self.matrix = cv2.imread(self.path)
```
In the second example, `self.matrix` is read from the file at initialization, which causes the computation to be done even-though the matrix is maybe not going to be used.

**Note**. `lazy_property` handling heavy or possibly changing objects should be used with parsimony. Remember that the object is kept in memory! If you want the property to be forgotten, please go for a method or use `del self.lazy_prop_name`.

3. Know the [basics of python optimization](https://stackify.com/20-simple-python-performance-tuning-tips/). If you doubt the efficiency of your code, use `commons.miscellaneous.timer` for robust experimenting.

### The code should be easily readable

#### Definition

Programmers with a reasonable level in python should be able to easily walk through the code, without struggling to find or understand things. This often conflicts with optimisation and memory management but should remain of paramount importance.

#### Examples

1. **Abbreviations** should be strictly prohibited unless absolutely necessary or blatantly clear ; They make no sense in the age of code completion.

 ❌ This is bad practice:
 ```python
def divs(n):
    ds = list()
    for i in range(n):
        if n % i == 0:
            ds.append(i)
    return ds
 ```

✅ This is much more readable:
```python
def find_divisors(dividend):
    divisors = []
    for candidate in range(dividend):
        if dividend % candidate == 0:
            divisors.append(candidate)
    return divisors
```

**Note**. A few tolerated extremely common abbreviations are:
- `img` for image
- `dir_` for directory
- `gt` for ground_truth
- `char` for character
- `vs` for variables
- If they can easily be understood from context (e.g. In loops and comprehensions : `[w for w in words]`)


2. **Variable names** should say what variables are

Just like *Ajax*'s name foreshadows his lament *Alas*, variable names should announce their value. Make names so eloquent you need the less comments possible : people can change your code but they very likely will not upgrade comments. A few examples :
- `image` should be an image, not the path to it.
- `file` should be an IO buffer, not a path or a filename.
- `number` should be an int or a float, `numbers` should be a collection
- By convention, `i` is used for an *iterator* itself, **not for the value** iterated upon. Please go for : `for thing in things` not `for i in things`
- Don't hesitate to use long names if they shed light on the code! `reordered_lines` is much better than `lines_2`.
- In any case avoid ambiguous, obscure or boilerplate names like `dict_new`, `dict_new_2`, `tmp_`, `values_list`, `to_output`...

3. **Functions names** should say what functions do.

Functions should be named with **verbs** which correspond to their actual action.

✅ This is good writing:
```python
def compute_rectangle_area(width, height):
    return width * height
```

❌ These are misleading
```python
# Get means you are fetching an already computed value
def get_rectangle_area(width, height):
    return width * height

# Substantives are kept for object
def rectangle_area_getter(width, height):
    return width * height
```

One exception to this principle are filters, i.e. function which test a condition and return a boolean:
```python
def is_square(rectangle):
    return rectangle.width == rectangle.height
```


4. Use simple objects and structure

Also adding to the readability of the code is its simplicity. Unless optimisation requires it, nested comprehension and smarter-than-smart structures should be avoided. Likewise, the use of special objects (i.e. non-native should be motivated). Use OOP, but remember that functional code is often easier to read. Objects should be objects at all only if there is a reason it.



## Other specific rules

1. Docstrings are written
   in [Google Style Formats](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
2. Architecture:
    - ⚠️ **scripts should be in scripts** ⚠️ Script files (i.e. files that actually do something when run) should always
      be located in each directory's dedicated `_scripts` directory, which should not be a python package itself (i.e.
      has no `__init__.py`).
    - `commons` should not import anything from task other dirs. `text_processing` can import only from commons.
      Task-specific dirs can import from everywhere.

3. Path management:
    - With the only exception of entrypoint function, paths are to be handled with `pathlib.Path` objects. "Entrypoints"
      are function that can be directly call by the API in a main pipeline; they should accept `Union[str, Path]` as
      input.
    - Path management is centralized in `ajmc.commons.variables`