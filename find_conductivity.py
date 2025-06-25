import os
from thermoml_fair.core.parser import parse_thermoml_xml
from thermoml_fair.core.schema import NumValuesRecord
import thermoml_fair

def get_data_file_path(filename: str) -> str:
    """Gets the absolute path to a data file included with the package."""
    package_dir = os.path.dirname(thermoml_fair.__file__)
    return os.path.join(package_dir, 'data', filename)

def find_thermal_conductivity_by_phase(xml_file_path: str, phase: str) -> list[NumValuesRecord]:
    """
    Parses a ThermoML XML file and returns records for thermal conductivity
    for a specific phase (e.g., "solid", "liquid").

    Args:
        xml_file_path: The absolute path to the ThermoML XML file.
        phase: The phase to filter for (case-insensitive).

    Returns:
        A list of NumValuesRecord objects that match the criteria.
    """
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"The file was not found at: {xml_file_path}")

    # Get the path to the ThermoML.xsd schema file
    schema_path = get_data_file_path('ThermoML.xsd')

    all_records, _ = parse_thermoml_xml(xml_file_path, schema_path)

    filtered_records = []
    for record in all_records:
        for prop_value in record.property_values:
            # Check for both the property name AND the phase
            is_correct_property = "Thermal conductivity" in prop_value.prop_name
            is_correct_phase = phase.lower() in prop_value.phase.lower()

            if is_correct_property and is_correct_phase:
                filtered_records.append(record)
                # This record matches, no need to check its other properties
                break
    
    return filtered_records

if __name__ == '__main__':
    # --- Example Usage ---
    # This example shows how to find thermal conductivity data for solids.
    
    try:
        # Get the example file that comes with the package
        example_file = get_data_file_path('j.tca.2007.01.009.xml')
        
        target_phase = "solid"
        print(f"Searching for thermal conductivity data for phase '{target_phase}' in: {example_file}\n")
        
        # Run the function to find the data for the specified phase
        results = find_thermal_conductivity_by_phase(example_file, phase=target_phase)
        
        if not results:
            print(f"No thermal conductivity data found for phase '{target_phase}' in this file.")
        else:
            print(f"Found {len(results)} records for thermal conductivity ({target_phase}):\n")
            for record in results:
                # Print some identifying information for each record
                if record.citation:
                    print(f"  - Journal: {record.citation.get('sPubName', 'N/A')}, Year: {record.citation.get('yrPubYr', 'N/A')}")
                for comp_name in record.components:
                    formula = record.compound_formulas.get(comp_name, "N/A")
                    print(f"    - Compound: {comp_name} ({formula})")
                for prop in record.property_values:
                    # Display only the matching properties from the record
                    if "Thermal conductivity" in prop.prop_name and target_phase.lower() in prop.phase.lower():
                        print(f"    - Property: {prop.prop_name}")
                        print(f"    - Phase: {prop.phase}")
                        for val in prop.values:
                             unit = prop.prop_name.split(',')[-1].strip()
                             print(f"      - Value: {val} {unit}")
                print("-" * 20)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have the 'thermoml-fair' package installed and that the example data files are available.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

