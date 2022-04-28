from jsonschema import Draft6Validator
import json

def validate_json_schema(schema_path:str = 'data/page.schema.json'):
    """Validates a json schema against `Draft6Validator`"""

    with open(schema_path, "r") as file:
        schema = json.loads(file.read())

    Draft6Validator.check_schema(schema)