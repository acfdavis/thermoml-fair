from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class VariableValue:
    var_type: str
    values: List[float]
    var_number: Optional[int] = None  # Existing field, ensure it's kept
    linked_component_org_num: Optional[int] = None # New field


@dataclass
class PropertyValue:
    prop_name: str
    values: List[float]
    uncertainties: List[Optional[str]]
    phase: str = ""  # New field for phase information
    method: str = ""  # New field for method name


@dataclass
class NumValuesRecord:
    material_id: str
    components: List[str]
    compound_formulas: Dict[str, str]
    variable_values: List[VariableValue]
    property_values: List[PropertyValue]
    component_id_map: Dict[int, str]  # in NumValuesRecord class
    source_file: str
    citation: Optional[Dict[str, str]] = None
