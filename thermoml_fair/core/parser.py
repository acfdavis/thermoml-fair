import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import xmlschema # Ensure xmlschema is imported
from collections import Counter


from thermoml_fair.core.schema import NumValuesRecord, VariableValue, PropertyValue

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_tag(d: dict, key: str, ns: str) -> Any:
    return d.get(key) or d.get(f"{ns}{key}") or d.get(key.replace(f"{ns}", ""))

def parse_thermoml_xml(file_path: str, xsd_path_or_obj: str | xmlschema.XMLSchema) -> Tuple[List[NumValuesRecord], List[Dict[str, Any]]]: # Modified type hint
    """
    Parse a ThermoML XML file into a list of NumValuesRecord instances.

    Parameters
    ----------
    file_path : str
        Path to the XML file.
    xsd_path_or_obj : str | xmlschema.XMLSchema
        Path to the XML Schema Definition (XSD) file for validation,
        or an already loaded xmlschema.XMLSchema object.

    Returns
    -------
    Tuple[List[NumValuesRecord], List[Dict[str, Any]]]
        A tuple containing:
        - List of parsed records with compounds, variables, and properties.
        - List of all compounds defined in the file, with their metadata.
    """


    if isinstance(xsd_path_or_obj, str):
        schema = xmlschema.XMLSchema(xsd_path_or_obj)
    elif isinstance(xsd_path_or_obj, xmlschema.XMLSchema):
        schema = xsd_path_or_obj
    else:
        raise TypeError("xsd_path_or_obj must be a string path or an xmlschema.XMLSchema object.")

    if not schema.is_valid(file_path):
        raise ValueError(f"{file_path} is not valid against the provided ThermoML schema.")

    data_dict = schema.to_dict(file_path)
    # Handle case where to_dict returns (data, errors)
    if isinstance(data_dict, tuple) and len(data_dict) == 2:
        data_dict, _ = data_dict

    ns = "{http://www.iupac.org/namespaces/ThermoML}"

    # Unwrap root
    if data_dict is not None and isinstance(data_dict, dict) and len(data_dict) == 1:
        data_dict = list(data_dict.values())[0]

    # Ensure data_dict is a dict before passing to get_tag
    if not isinstance(data_dict, dict):
        logger.error("Parsed XML root is not a dictionary. Cannot continue parsing.")
        return [], []

    compounds_section = get_tag(data_dict, "Compound", ns) or []
    pure_data_section = get_tag(data_dict, "PureOrMixtureData", ns) or []
    if not isinstance(pure_data_section, list):
        pure_data_section = [pure_data_section]

    compound_map = {}
    all_compounds = []
    for compound in compounds_section:
        try:
            org_num = int(get_tag(get_tag(compound, "RegNum", ns), "nOrgNum", ns))
            
            compound_details: Dict[str, Any] = {"nOrgNum": org_num}

            # Extract all fields from the compound, handling lists
            for key in ["sCommonName", "sFormulaMolec", "sCASName", "sCID", "sCASRegistryNum", "sInChI", "sInChIKey", "sSmiles"]:
                value_entry = get_tag(compound, key, ns)
                if value_entry:
                    value = value_entry[0] if isinstance(value_entry, list) else value_entry
                    if isinstance(value, str):
                        compound_details[key] = value.strip()

            name = compound_details.get("sCommonName", "")
            formula = compound_details.get("sFormulaMolec", "")

            compound_map[org_num] = {"name": name, "formula": formula}
            all_compounds.append(compound_details)
        except Exception as e:
            logger.warning(f"Skipping invalid compound: {e}")
    
    citation = {}
    citation_section = get_tag(data_dict, "Citation", ns)
    if citation_section:
        for tag in ["sDOI", "sTitle", "sPubName", "yrPubYr", "sAuthor"]:
            value = get_tag(citation_section, tag, ns)
            if value is not None:
                citation[tag] = value.strip() if isinstance(value, str) else value
        
    results = []
    for entry in pure_data_section:
        try:
            component_ids = [
                str(get_tag(get_tag(c, "RegNum", ns), "nOrgNum", ns))
                for c in get_tag(entry, "Component", ns) or []
            ]
            material_id = "__".join(sorted(component_ids)) if component_ids else "unknown"
         
            components = []
            compound_formulas = {}
            component_id_map = {} 

            for comp in get_tag(entry, "Component", ns) or []:
                regnum = get_tag(comp, "RegNum", ns)
                if regnum is None or not isinstance(regnum, dict):
                    logger.warning(f"Invalid RegNum: {regnum}")
                    continue
                org_num = int(get_tag(regnum, "nOrgNum", ns))
                info = compound_map.get(org_num, {})
                name = info.get("name", f"Unknown-{org_num}")
                components.append(name)
                compound_formulas[name] = info.get("formula", "")
                component_id_map[org_num] = info.get("formula") if info.get("formula") else name  # Track for matching with var_number

            prop_name_map = {}
            prop_phase_map = {}
            prop_method_map = {}
            for prop in get_tag(entry, "Property", ns) or []:
                try:
                    num = int(get_tag(prop, "nPropNumber", ns))
                    group = get_tag(get_tag(prop, "Property-MethodID", ns), "PropertyGroup", ns)                    
                    group = (
                        get_tag(group, "VolumetricProp", ns)
                        or get_tag(group, "TransportProp", ns)
                        or get_tag(group, "ThermodynProp", ns)
                        or {}
                    )
                    name = get_tag(group, "ePropName", ns)
                    method = get_tag(group, "eMethodName", ns) or ""
                    phase_entry = get_tag(prop, "PropPhaseID", ns)
                    phase = ""
                    if isinstance(phase_entry, list) and phase_entry:
                        phase = get_tag(phase_entry[0], "ePropPhase", ns)
                    prop_name_map[num] = name or "unknown"
                    prop_phase_map[num] = phase
                    prop_method_map[num] = method
                except Exception as e:
                    # logger.debug(f"Skipping property due to: {e}")
                    continue

            var_type_map = {}
            var_def_to_comp_orgnum_map = {}  # New map: var_def_id -> linked_org_num

            for var_def in get_tag(entry, "Variable", ns) or []: # Iterate over <Variable> definitions
                try:
                    num = int(get_tag(var_def, "nVarNumber", ns)) # This is the ID of the variable definition
                    variable_id_block = get_tag(var_def, "VariableID", ns)
                    vtype_entry = get_tag(variable_id_block, "VariableType", ns)
                    
                    if isinstance(vtype_entry, dict):
                        vtype = vtype_entry.get("eVarType") or vtype_entry.get("e") or list(vtype_entry.values())[0]
                    elif isinstance(vtype_entry, str):
                        vtype = vtype_entry
                    else:
                        vtype = "UnknownType"
                   
                    var_type_map[num] = vtype

                    # Check for linked component in VariableID
                    comp_regnum_block = get_tag(variable_id_block, "RegNum", ns)
                    if comp_regnum_block and isinstance(comp_regnum_block, dict):
                        linked_org_num_str = get_tag(comp_regnum_block, "nOrgNum", ns)
                        if linked_org_num_str is not None:
                            var_def_to_comp_orgnum_map[num] = int(linked_org_num_str)
                            
                except Exception as e:
                    # logger.debug(f"Skipping variable definition due to: {e}")
                    continue
            
            # variable_values = [] # This line is part of existing code, ensure it's placed correctly relative to the loop below
            # property_values = [] # This line is part of existing code
            num_values = get_tag(entry, "NumValues", ns) or []
            num_values = num_values if isinstance(num_values, list) else [num_values]
           
            # Count occurrences of each variable type
            vtype_counts = Counter(var_type_map.values())

            for nv in num_values:
                # Reset lists for each NumValues block if they are supposed to be specific to it
                # If variable_values and property_values are cumulative for the entry, initialize them before this loop.
                # Based on current structure, they seem cumulative for the NumValues block, then a record is made.
                # Let's assume they are re-initialized for each record that will be appended.
                current_variable_values = []
                current_property_values = []

                for vv in get_tag(nv, "VariableValue", ns) or []:
                    try:
                        var_def_id = int(get_tag(vv, "nVarNumber", ns))
                        var_value_str = get_tag(vv, "nVarValue", ns) # Keep as string initially for logging
                        if var_value_str is None:
                            # logger.debug(f"Skipping VariableValue with var_def_id {var_def_id}: nVarValue is None.")
                            continue
                        
                        var_value = float(var_value_str) # Convert to float after ensuring it's not None
                        
                        vtype = var_type_map.get(var_def_id, "")
                        # Use a more descriptive label if multiple variables have the same type
                        # This logic was already present, just ensuring it's clear
                        if vtype_counts[vtype] > 1:
                            var_label = f"{vtype}_{var_def_id}"
                        else:
                            var_label = vtype
                        
                        actual_linked_org_num = var_def_to_comp_orgnum_map.get(var_def_id)
                        
                        # Log the details of the VariableValue being created
                        # logger.debug(
                        #     f"Processing VariableValue: var_def_id={var_def_id}, "
                        #     f"raw_nVarValue='{var_value_str}', parsed_float_value={var_value}, "
                        #     f"vtype='{vtype}', final_var_label='{var_label}', "
                        #     f"linked_component_org_num={actual_linked_org_num}"
                        # )

                        current_variable_values.append(VariableValue(
                            var_type=var_label,
                            values=[var_value],
                            var_number=var_def_id,
                            linked_component_org_num=actual_linked_org_num
                        ))
                    except Exception as e:
                        # logger.debug(f"Skipping VariableValue due to: {e} (raw data: {get_tag(vv, 'nVarValue', ns)})")
                        continue

                for pv in get_tag(nv, "PropertyValue", ns) or []:
                    try:
                        prop_number = int(get_tag(pv, "nPropNumber", ns))
                        prop_value = float(get_tag(pv, "nPropValue", ns))

                        uncertainty = None
                        u_block = get_tag(pv, "PropUncertainty", ns)
                        if isinstance(u_block, list) and u_block:
                            uncertainty = float(get_tag(u_block[0], "nStdUncertValue", ns) or 0.0)
                        elif isinstance(u_block, dict):
                            uncertainty = float(get_tag(u_block, "nStdUncertValue", ns) or 0.0)

                        # Get phase and method for this property
                        phase = prop_phase_map.get(prop_number, "")
                        method = prop_method_map.get(prop_number, "")

                        current_property_values.append(PropertyValue(
                            prop_name=prop_name_map.get(prop_number, "unknown"),
                            values=[prop_value],
                            uncertainties=[str(uncertainty)] if uncertainty is not None else [],
                            phase=phase,
                            method=method
                        ))
                    except Exception as e:
                        # logger.debug(f"Skipping PropertyValue due to: {e}")
                        continue

                # Create record after processing all VariableValues and PropertyValues for this NumValues block (nv)
                # Log the collected variable values before creating the record
                # logger.debug(f"Finalizing NumValuesRecord for material_id: {material_id} with {len(current_variable_values)} VariableValues and {len(current_property_values)} PropertyValues.")
                # for cv_idx, cv_val in enumerate(current_variable_values):
                #     logger.debug(
                #         f"  Record's VariableValue {cv_idx}: "
                #         f"var_type='{cv_val.var_type}', "
                #         f"values={cv_val.values}, "
                #         f"var_number={cv_val.var_number}, "
                #         f"linked_component_org_num={cv_val.linked_component_org_num}"
                #     )
                if current_variable_values or current_property_values: # Only create a record if there's data
                    record = NumValuesRecord(
                        material_id=str(material_id),
                        components=components, # List of component names
                        compound_formulas=compound_formulas, # Dict: name -> formula (symbol)
                        variable_values=current_variable_values,
                        property_values=current_property_values,
                        component_id_map=component_id_map, # Dict: nOrgNum -> formula (symbol)
                        source_file=file_path,
                        citation=citation
                    )
                    results.append(record)

        except Exception as e:
            logger.warning(f"Skipping entry due to error: {e}")
            continue

    return results, all_compounds
