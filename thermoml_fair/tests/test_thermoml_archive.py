import os
from pathlib import Path
import xmlschema

def find_archive_dir():
    return Path(os.environ.get("THERMOML_PATH", Path.home() / ".thermoml"))

def list_xml_files(directory):
    return sorted(directory.glob("*.xml"))

def validate_with_schema(xml_files, schema_path, limit=10):
    schema = xmlschema.XMLSchema(schema_path)
    for xml_file in xml_files[:limit]:
        try:
            schema.validate(xml_file)
            print(f"✅ Valid: {xml_file.name}")
        except Exception as e:
            print(f"❌ Invalid: {xml_file.name} — {e}")

if __name__ == "__main__":
    archive_path = find_archive_dir()
    print(f"Checking archive directory: {archive_path}")

    xml_files = list_xml_files(archive_path)
    print(f"Found {len(xml_files)} XML files.")

    # Use your specified schema path
    schema_file = Path(__file__).parent.parent / "data" / "ThermoML.xsd"
    if schema_file.exists():
        print(f"Validating up to 10 XML files with schema: {schema_file}")
        validate_with_schema(xml_files, schema_file)
    else:
        print(f"⚠️ Schema file not found at {schema_file} — skipping validation.")
