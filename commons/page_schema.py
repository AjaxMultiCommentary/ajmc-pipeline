# This contains the changes made to the impresso-schema
# impresso-text-acquisition/src/impresso-commons/impresso_commons/schemas/json/newspaper/page.schema.json
changes = {
    "cc":None,
    "cdt": "cdate",
    "r": "regions",
    "pOf": None,
    "c": "coords",
    "l":"lines",
    "t":"words",
    "s":None,
    "gn":None,
    "hy":None,
    "nf":None,

}

# Possible mentions add :
# - word and lines confidence
# todo: change "$id" later on
# todo: see how to handle the path to images




from jsonschema import Draft6Validator
import json
schema_path = "/commons/page.schema.json"
# schema_path = "/Users/sven/packages/impresso/impresso-text-acquisition/src/impresso-commons/impresso_commons/schemas/json/newspaper/page.schema.json"
with open(schema_path, "r") as file:
    schema = json.loads(file.read())

Draft6Validator.check_schema(schema)