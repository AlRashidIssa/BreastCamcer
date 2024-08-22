import json

def save_schema_to_json(schema: dict)-> None:
    """
    
    
    """
    # save schema to a file
    with open("model_schema.json", 'w') as f:
        json.dump(schema, f)