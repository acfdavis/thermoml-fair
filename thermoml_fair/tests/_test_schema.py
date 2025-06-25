import os
import xmlschema

def validate_xml(schema_path, file_path):
    schema = xmlschema.XMLSchema(schema_path)
    schema.validate(file_path)

def test_all_thermoml_files_validate():
    schema_path = "ThermoML.xsd"
    data_dir = "thermoml-fair/data"
    xml_files = [f for f in os.listdir(data_dir) if f.endswith(".xml")]

    assert xml_files, "No XML files found for validation."

    errors = []
    for xml_file in xml_files:
        full_path = os.path.join(data_dir, xml_file)
        try:
            validate_xml(schema_path, full_path)
        except xmlschema.XMLSchemaValidationError as e:
            errors.append(f"{xml_file} failed: {e}")

    assert not errors, "Schema validation failed:\n" + "\n".join(errors)
