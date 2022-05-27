"""This file contains generic docstring chunks to be formatted using `docstring_formatter`."""

docstrings = {
    'custom_dataset':
        """A dataset inheriting from `torch.utils.data.Dataset`, implementing at least `__len__` and 
        `__getitem__()`, where each item is a dict alike `{{"model_input": tensor(), ...}}` corresponding
        to a single example.""",

    'do_debug':
        """Whether break loops after the first iteration.""",

    'ids_to_labels':
        """A dict mapping the label numbers (int) used by the model to the original 
        label names (str), e.g. `{{0: "O", 1: "B-PERS", ...}}`""",

    'labels_to_ids': 'A dict mapping label-names to their respective ids, e.g. `{{"cat":0, "dog":1, ...}}`.',

    'max_length': 'The maximum length of a sequence to be processed by the model.',

}
