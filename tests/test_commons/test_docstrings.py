from ajmc.commons.docstrings import docstring_formatter


def test_docstring_formatter():
    @docstring_formatter(alpha=1, beta=2)
    def f():
        """{alpha} {beta}"""
        pass

    assert f.__doc__ == '1 2'
