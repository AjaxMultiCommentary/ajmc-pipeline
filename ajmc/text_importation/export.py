import os
from typing import List
from jinja2 import Environment, FileSystemLoader


def page_to_alto(page: 'TextContainer',
                 elements: List[str],
                 output_path: str):
    """Exports a page to ALTO-xml.

    Args:
        page: A text container representing a page.
        elements: The list of sub-page element-types you want to includ, e.g. `['region', 'line']`.
        output_path: self-explanatory.
    """
    file_loader = FileSystemLoader('data/templates')
    env = Environment(loader=file_loader)
    template = env.get_template('alto.xml.jinja2')

    with open(output_path, 'w') as f:
        f.write(template.render(page=page, elements=elements))


def commentary_to_alto(commentary: 'TextContainer',
                       elements: List[str],
                       output_dir: str):
    """Uses `page_to_alto` on an entire commentary. """

    for p in commentary.children['page']:
        page_to_alto(page=p, elements=elements, output_path=os.path.join(output_dir, p.id + '.xml'))
