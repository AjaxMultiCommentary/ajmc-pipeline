"""This file contains generic docstring chunks to be formatted using `docstring_formatter`."""


def docstring_formatter(**kwargs):
    # Todo : One could also add automatic typing, if necessary (i.e. taking type hints directly from params.
    """Decorator with arguments used to format the docstring of a functions.

    `docstring_formatter` is a decorator with arguments, which means that it takes any set of `kwargs` as argument and
    returns a decorator. It should therefore always be called with parentheses (unlike traditional decorators - see
    below). It follows the grammar of `str.format()`, i.e. `{my_format_value}`.
    grammar.

    Example:
        For instance, this code :

        ```Python
        @docstring_formatter(greeting = 'hello')
        def my_func():
            "A simple greeter that says {greeting}"
            # Do your stuff
        ```

        Is actually equivalent with :

        ```Python
        def my_func():
            "A simple greeter that says {greeting}"
            # Do your stuff

        my_func.__doc__ = my_func.__doc__.format(greeting = 'hello')
        ```
    """

    def inner_decorator(func):
        func.__doc__ = func.__doc__.format(**kwargs)
        return func

    return inner_decorator


docstrings = dict()  # Creating docstrings on the fly in order to refer to previously declared elements.

docstrings['BatchEncoding'] = """The default ouput of HuggingFace's `TokenizerFast`. As [docs](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/tokenizer#transformers.BatchEncoding)
        have it, "This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes utility methods to map from word/character space to token space". 
        The object contains `data` and `encodings`. Data is directly callable and has the form of a `Dict[str, List[List[int]]]` where keys are model inputs. Encodings is a 
        list of example, containing notably the offsets.""",

docstrings['custom_dataset'] = """A dataset inheriting from `torch.utils.data.Dataset`, implementing at least `__len__` and 
        `__getitem__()`, where each item is a dict alike `{{"model_input": tensor(), ...}}` corresponding
        to a single example.""",

docstrings['do_debug'] = """Whether break loops after the first iteration.""",

docstrings['ids_to_labels'] = """A dict mapping the label numbers (int) used by the model to the original label names (str), e.g. `{{0: "O", 1: "B-PERS", ...}}`""",

docstrings['labels_to_ids'] = 'A dict mapping label-names to their respective ids, e.g. `{{"cat":0, "dog":1, ...}}`.',

docstrings['transformers_model'] = """A `transformers.models`."""

docstrings['transformers_model_inputs'] = """A mapping to between the names of the model's requirements and `torch.Tensor` of size (max_length, batch_size).
    Example : `{'input_ids': torch.tensor([[int, int, ...], [int, int, ...]])`."""

docstrings['transformers_model_predictions'] = """`np.ndarray` containing the predicted labels, so in the shape (number of exs, length of an ex)."""

docstrings['max_length'] = 'The maximum length of a sequence to be processed by the model.',

docstrings['special_tokens'] = """LEGACY. A dict containing the model's special token for sequence start, end and pad. 
e.g. `{{'start': {{'input_ids':100, ...}}, ...}}`"""

docstrings['via_project'] = """A dictionary resulting from the reading of a via_project JSON file. Visit 
https://www.robots.ox.ac.uk/~vgg/software/via/ for more information"""