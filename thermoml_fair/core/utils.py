import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union # Added Union
from pathlib import Path # Added
import xmlschema # Added for type hinting
import traceback # Added
from thermoml_fair.core.parser import parse_thermoml_xml
from pymatgen.core import Element, Composition # Ensure Composition is imported
import logging
from thermoml_fair import __version__ as thermoml_fair_version # Added import for version
from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, as_completed
from thermoml_fair.core.config import THERMOML_PATH, DEFAULT_CACHE_DIR_NAME

logger = logging.getLogger(__name__)

def get_fn(filename: str) -> str:
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    return filename  # fallback if running locally

def get_cache_dir() -> Path:
    """Return the default cache directory path."""
    return Path(os.path.expanduser(THERMOML_PATH)) / DEFAULT_CACHE_DIR_NAME

# Helper function to check pymatgen availability
def _is_pymatgen_available() -> bool:
    try:
        # Check for Element and Composition classes
        if not (callable(getattr(Element, "is_valid_symbol", None)) and hasattr(Element("H"), "atomic_mass") and callable(getattr(Composition, "get_el_amt_dict", None))):
            return False
        return True
    except ImportError:
        return False
    except Exception: # Other potential issues during check
        return False

def load_repository_metadata(metadata_path: str = None) -> dict:
    """
    Load repository-level metadata from archive_info.json (NERDm metadata).
    If metadata_path is None, will look for ~/.thermoml/archive_info.json.
    """
    if metadata_path is None:
        home = os.path.expanduser("~")
        metadata_path = os.path.join(home, ".thermoml", "archive_info.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.warning(f"Could not load repository metadata from {metadata_path}: {e}")
        return {}

def parse_one(file_path, xsd_path_or_obj, failed_files=None):
    schema_for_parsing: Union[str, xmlschema.XMLSchema]
    if xsd_path_or_obj is not None:
        if isinstance(xsd_path_or_obj, Path):
            schema_for_parsing = str(xsd_path_or_obj)
        else:
            schema_for_parsing = xsd_path_or_obj
    else:
        schema_for_parsing = os.path.join(os.path.dirname(__file__), "..", "data", "ThermoML.xsd")
    try:
        return parse_thermoml_xml(file_path, xsd_path_or_obj=schema_for_parsing)
    except Exception as e:
        if failed_files is not None:
            failed_files.append(file_path)
        # logger.error(f"Failed to parse XML file {file_path} with schema {schema_for_parsing}: {e}")
        return [], []


def build_pandas_dataframe(
    xml_files: List[str], 
    normalize_alloys: bool = False, 
    repository_metadata: Optional[dict] = None,
    xsd_path_or_obj: Optional[Union[str, Path, xmlschema.XMLSchema]] = None, # Added new parameter
    max_workers: Optional[int] = None, # New parameter for parallelism
    show_failed_files: bool = False # New parameter for optional detailed failed file list
) -> dict:
    """
    Build pandas DataFrames from ThermoML XML files and include repository-level metadata.
    Returns a dict with keys: 'data', 'compounds', 'repository_metadata'.
    Now uses parallel processing for XML parsing.
    """
    if repository_metadata is None:
        repository_metadata = load_repository_metadata()

    all_records = []
    all_compounds_data = []
    unique_properties = set()
    failed_files = []

    # Get thermoml_fair version
    current_thermoml_fair_version = thermoml_fair_version

    def pretty_formula(frac_dict: Dict[str, float]) -> str:
        parts = []
        significant_fracs = {el: amt for el, amt in frac_dict.items() if amt > 1e-5}
        if not significant_fracs:
            return ""
        for el, amt in sorted(significant_fracs.items()):
            if abs(amt - 1.0) < 1e-3 and len(significant_fracs) == 1:
                parts.append(f"{el}")
            else:
                parts.append(f"{el}{round(amt, 3)}")
        return "".join(parts)

    # Use parallel processing for parsing
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]
    results = []
    if max_workers == 1:
        # Single-worker (test/mocking) path: no parallelism, but same output structure
        with Progress(*progress_columns) as progress:
            task_id = progress.add_task("Parsing XML files...", total=len(xml_files))
            for file_path in xml_files:
                parsed_records, parsed_compounds = parse_one(file_path, xsd_path_or_obj, failed_files=failed_files)
                results.append((file_path, (parsed_records, parsed_compounds)))
                progress.update(task_id, advance=1)
    else:
        with Progress(*progress_columns) as progress:
            task_id = progress.add_task("Parsing XML files...", total=len(xml_files))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(parse_one, file_path, xsd_path_or_obj, failed_files): file_path for file_path in xml_files}
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    parsed_records, parsed_compounds = future.result()
                    results.append((file_path, (parsed_records, parsed_compounds)))
                    progress.update(task_id, advance=1)

    # Flatten results and process as before
    all_parsed_records = []
    for file_path, (parsed_records, parsed_compounds) in results:
        all_parsed_records.extend(parsed_records)
        for compound in parsed_compounds:
            compound['source_file'] = file_path
        all_compounds_data.extend(parsed_compounds)

    for record in all_parsed_records:
        for prop in record.property_values:
            unique_properties.add(prop.prop_name)
            row: Dict[str, Any] = {
                "material_id": record.material_id,
                "components": ", ".join(record.components),
                "thermoml_fair_version": current_thermoml_fair_version,
                "property": prop.prop_name,
                "value": prop.values[0] if prop.values else None,
                "phase": prop.phase or "",
                "method": prop.method or "",
                "uncertainty": prop.uncertainties[0] if prop.uncertainties else None,
                "source_file": record.source_file,
            }
            # Add citation info
            if record.citation:
                row["doi"] = record.citation.get("sDOI")
                row["publication_year"] = row["publication_year"] = record.citation.get("yrPubYr")
                row["title"] = record.citation.get("sTitle")
                s_authors_list = record.citation.get("sAuthor")
                if s_authors_list and isinstance(s_authors_list, list) and len(s_authors_list) > 0:
                    first_author_full_string = s_authors_list[0]
                    row["author"] = first_author_full_string.split(';')[0].strip()
                else:
                    row["author"] = None
                row["journal"] = record.citation.get("sPubName")
            else:
                row["doi"] = None
                row["publication_year"] = None
                row["title"] = None
                row["author"] = None
                row["journal"] = None
            # Convert None values in citation fields to empty strings for DataFrame consistency
            citation_keys = ["doi", "publication_year", "title", "author", "journal"]
            for key in citation_keys:
                if row[key] is None:
                    row[key] = ""
            all_records.append(row)

    # Create DataFrames from the collected records
    df = pd.DataFrame(all_records)

    # Ensure normalized_formula and active_components columns exist if normalize_alloys is True
    if normalize_alloys:
        if "normalized_formula" not in df.columns:
            df["normalized_formula"] = ""
        if "active_components" not in df.columns:
            df["active_components"] = ""

    # If alloy normalization is requested, fill normalized_formula and active_components for each row
    if normalize_alloys and not df.empty:
        rec_map = {rec.material_id: rec for rec in all_parsed_records}
        def build_normalized_formula(row):
            mat_id = row.get("material_id", None)
            rec = rec_map.get(mat_id, None)
            if rec is None or not hasattr(rec, "components"):
                comps = row.get("components", "")
                return ''.join([c.strip() for c in comps.split(',') if c.strip()])
            orgnum_to_symbol = {k: v for k, v in getattr(rec, 'component_id_map', {}).items()}
            all_symbols = list(orgnum_to_symbol.values())
            fractions = {}
            for var in getattr(rec, 'variable_values', []):
                if var.var_type and ("mole fraction" in var.var_type.lower() or "mass fraction" in var.var_type.lower()):
                    if var.linked_component_org_num is not None and var.values:
                        symbol = orgnum_to_symbol.get(var.linked_component_org_num)
                        if symbol:
                            fractions[symbol] = var.values[0]
            # Fill missing fractions if only one is missing
            missing = [s for s in all_symbols if s not in fractions]
            if len(missing) == 1 and len(fractions) > 0:
                fractions[missing[0]] = max(0.0, 1.0 - sum(fractions.values()))
            # Always include all components in the formula
            if fractions and abs(sum(fractions.get(s, 0.0) for s in all_symbols) - 1.0) < 0.05:
                return pretty_formula({s: fractions.get(s, 0.0) for s in all_symbols})
            # fallback: join all symbols
            return ''.join([s for s in all_symbols if s])
        def build_active_components(row):
            mat_id = row.get("material_id", None)
            rec = rec_map.get(mat_id, None)
            if rec is None or not hasattr(rec, "components"):
                comps = row.get("components", "")
                return ', '.join([c.strip() for c in comps.split(',') if c.strip()])
            orgnum_to_symbol = {k: v for k, v in getattr(rec, 'component_id_map', {}).items()}
            comps = [orgnum_to_symbol.get(org, org) for org in orgnum_to_symbol]
            return ', '.join([c for c in comps if c])
        df['normalized_formula'] = df.apply(build_normalized_formula, axis=1)
        df['active_components'] = df.apply(build_active_components, axis=1)

    # Create compounds DataFrame
    compounds_df = pd.DataFrame(all_compounds_data)
    
    # Add normalized_formula to compounds_df if normalize_alloys is True
    if normalize_alloys:
        if _is_pymatgen_available():
            def normalize_formula(formula):
                if not formula or not isinstance(formula, str):
                    return ''
                try:
                    comp = Composition(formula)
                    return comp.get_reduced_formula_and_factor()[0]
                except Exception:
                    return formula
            
            if 'sFormulaMolec' in compounds_df.columns:
                compounds_df['normalized_formula'] = compounds_df['sFormulaMolec'].apply(normalize_formula)
            else:
                compounds_df['normalized_formula'] = ''
        else:
            compounds_df['normalized_formula'] = ''

    if not compounds_df.empty:
        # Drop duplicates based on a unique identifier
        subset_keys = ['sInChIKey']
        if 'sStandardInChIKey' not in compounds_df.columns or compounds_df['sStandardInChIKey'].isnull().all():
            subset_keys = ['sCommonName', 'sFormulaMolec', 'sCASRegistryNum']
            # Ensure fallback keys exist before using them for deduplication
            for key in subset_keys:
                if key not in compounds_df.columns:
                    compounds_df[key] = '' # Add missing key as empty column
        else:
            subset_keys = ['sStandardInChIKey']

        # Final check on subset_keys before dropping duplicates
        valid_subset_keys = [k for k in subset_keys if k in compounds_df.columns]
        if valid_subset_keys:
            compounds_df = compounds_df.drop_duplicates(subset=valid_subset_keys).reset_index(drop=True)

    # Create properties DataFrame
    properties_df = pd.DataFrame(sorted(list(unique_properties)), columns=['sPropName'])

    # After parsing, print summary of failed files
    if failed_files:
        print(f"\n⚠️ Failed to parse {len(failed_files)} XML file(s) out of {len(xml_files)}.")
        if show_failed_files:
            print("Failed files:")
            for f in failed_files:
                print(f"  - {f}")

    return {
        "data": df,
        "compounds": compounds_df,
        "properties": properties_df,
        "repository_metadata": repository_metadata
    }


def pandas_dataframe(path: str) -> pd.DataFrame:
    try:
        result = pd.read_hdf(os.path.join(path, "data.h5"), key="data")
        if isinstance(result, pd.Series):
            result = result.to_frame().T
        return result
    except Exception as e:
        print(f"Failed to load DataFrame from HDF5: {e}")
        return pd.DataFrame()
