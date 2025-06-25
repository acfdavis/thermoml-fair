import typer
from pathlib import Path
import os
import logging
import pickle
import json
import xmlschema # For schema validation and parsing
from typing import List, Optional, Set
import dataclasses # Added for dataclass handling
import pandas as pd
from rich.console import Console
from rich.table import Table
import traceback # For more detailed error reporting

from thermopyl import __version__
from thermopyl.core.parser import parse_thermoml_xml
from thermopyl.core.utils import build_pandas_dataframe
from thermopyl.core.update_archive import update_archive as update_archive_core
from thermopyl.core.config import THERMOML_PATH, THERMOML_SCHEMA_PATH # Import default schema path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define SUPPORTED_DATAFRAME_FORMATS at the module level
SUPPORTED_DATAFRAME_FORMATS: Set[str] = {".csv", ".h5", ".hdf5", ".parquet"}

app = typer.Typer(name="thermopyl", help="ThermoML data processing toolkit.")
console = Console()
# Helper function to get schema path
# DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "data" / "ThermoML.xsd" # Replaced by THERMOML_SCHEMA_PATH

# Helper function to convert various object types to dictionaries for pickling
def _to_dict_for_pickle(obj: object) -> dict:
    """
    Converts an object to a dictionary suitable for pickling.
    Handles standard objects, dataclasses, namedtuples, and __slots__-based objects.
    """
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    # Check for NamedTuple specifically or tuple for broader compatibility if needed
    if isinstance(obj, tuple) and hasattr(obj, '_asdict'):  # For NamedTuple
        return obj._asdict() # type: ignore[attr-defined]
    if hasattr(obj, '__dict__'): # For standard class instances
        return vars(obj)
    if hasattr(obj, '__slots__'): # For classes with __slots__
        return {slot: getattr(obj, slot) for slot in obj.__slots__ if hasattr(obj, slot)} # type: ignore[attr-defined]
    
    # Fallback or error for unhandled types
    logger.warning(f"Object of type {type(obj)} could not be automatically converted to a dict. Pickling may be incomplete or fail.")
    # The original code's `dict(obj)` failed for `NumValuesRecord`, so avoid repeating that directly without context.
    # If this path is reached, it implies an object type not covered by common patterns.
    # Depending on strictness, could raise TypeError or attempt `dict(obj)` with caution.
    # For now, returning an empty dict with a warning, or raising an error, are options.
    # Raising an error is safer to highlight unhandled cases.
    raise TypeError(f"Cannot convert object of type {type(obj)} to dict for pickling using known methods.")


def get_schema_object() -> Optional[xmlschema.XMLSchema]:
    """
    Loads and returns the XMLSchema object.
    Priority for path:
    1. THERMOPYL_SCHEMA_PATH environment variable.
    2. Default schema path from thermopyl.core.config.THERMOML_SCHEMA_PATH.
    Returns None if the schema cannot be loaded.
    """
    env_schema_path_str = os.environ.get("THERMOPYL_SCHEMA_PATH")
    schema_to_load_path: Optional[Path] = None

    if env_schema_path_str:
        env_schema_file = Path(env_schema_path_str)
        if env_schema_file.is_file():
            logger.info(f"Using schema from environment variable: {env_schema_file}")
            schema_to_load_path = env_schema_file
        else:
            logger.warning(
                f"Schema path from THERMOPYL_SCHEMA_PATH not found: {env_schema_file}. "
                "Falling back to default schema."
            )

    if schema_to_load_path is None: # If env var not set or path invalid
        if THERMOML_SCHEMA_PATH and Path(THERMOML_SCHEMA_PATH).is_file():
            logger.info(f"Using default schema: {THERMOML_SCHEMA_PATH}")
            schema_to_load_path = Path(THERMOML_SCHEMA_PATH)
        else:
            logger.error(f"Default schema not found at {THERMOML_SCHEMA_PATH}. Please ensure it's correctly installed or set THERMOPYL_SCHEMA_PATH.")
            return None
    
    if schema_to_load_path:
        try:
            return xmlschema.XMLSchema(str(schema_to_load_path))
        except Exception as e:
            logger.error(f"Failed to load XMLSchema object from {schema_to_load_path}: {e}")
            typer.echo(f"[ERROR] Failed to load schema from {schema_to_load_path}: {e}", err=True)
            typer.echo(traceback.format_exc(), err=True)
            return None
    return None


@app.command()
def version():
    """Show the version of thermopyl."""
    typer.echo(f"thermopyl version: {__version__}")


@app.command()
def parse(
    file: Path = typer.Option(..., "--file", "-f", help="Path to ThermoML XML file", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save .parsed.pkl file. Defaults to the input file's directory.", file_okay=False, dir_okay=True, writable=True),
    overwrite: bool = typer.Option(False, "--overwrite", "-ow", help="Overwrite existing .pkl file.")
):
    """
    Parse a single ThermoML XML file and save the result as a .pkl file.
    The .pkl file contains a list of ThermoMLRecord-like dictionaries.
    """
    typer.echo(f"Parsing ThermoML file: {file}")
    
    schema_obj = get_schema_object()
    if schema_obj is None:
        typer.echo("[ERROR] Schema could not be loaded. Cannot parse the file.", err=True)
        raise typer.Exit(code=2)

    if output_dir is None:
        output_dir = file.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    output_pkl_path = output_dir / file.with_suffix(".parsed.pkl").name

    if output_pkl_path.exists() and not overwrite:
        typer.echo(f"Output file {output_pkl_path} already exists. Use --overwrite to replace it.")
        raise typer.Exit(code=1)

    try:
        # Pass the loaded schema object to parse_thermoml_xml
        records = parse_thermoml_xml(str(file), schema_obj)
        
        # Convert records to dictionaries for pickling
        record_dicts = [_to_dict_for_pickle(r) for r in records]

        with open(output_pkl_path, "wb") as f:
            pickle.dump(record_dicts, f)
        typer.echo(f"Successfully parsed {len(records)} records from {file}.")
        typer.echo(f"Saved parsed data to: {output_pkl_path}")
        if records:
            typer.echo("\\nFirst few records (preview):") # Corrected string escaping
            for record_dict in record_dicts[:3]:
                # Basic preview, can be enhanced
                doi = record_dict.get('citation', {}).get('sDOI', 'N/A') if isinstance(record_dict.get('citation'), dict) else 'N/A'
                components_len = len(record_dict.get('components', []))
                typer.echo(f"  DOI: {doi}, Components: {components_len}")


    except ValueError as e: # Catch schema validation errors or parsing errors from parse_thermoml_xml
        typer.echo(f"[ERROR] Processing Error for {file}: {e}", err=True)
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.echo(f"[ERROR] Failed to parse {file}: {e}", err=True)
        typer.echo(traceback.format_exc(), err=True) # Print full traceback for debugging
        raise typer.Exit(code=2)


@app.command()
def validate(
    file: Path = typer.Option(..., "--file", "-f", help="Path to ThermoML XML file", file_okay=True, dir_okay=False, readable=True),
):
    """Validate a ThermoML XML file against the schema."""
    if not file.exists():
        msg = f"File not found: {file}"
        typer.echo(msg)
        typer.echo(msg, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"Validating {file}...")
    
    schema_obj = get_schema_object()
    if schema_obj is None:
        typer.echo("[ERROR] Schema could not be loaded. Cannot validate the file.", err=True)
        raise typer.Exit(code=2)

    try:
        if schema_obj.is_valid(str(file)):
            typer.echo(f"{file} is valid against the loaded ThermoML schema.")
        else:
            typer.echo(f"{file} is NOT valid against the loaded ThermoML schema.", err=True)
            # Attempt to get more detailed errors
            errors = list(schema_obj.iter_errors(str(file)))
            if errors:
                typer.echo("Validation errors:", err=True)
                for error in errors[:5]: # Show first 5 errors
                    typer.echo(f"  - {error.message} (Path: {error.path})", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during validation: {e}", err=True)
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=2)

@app.command()
def parse_all(
    input_dir: Path = typer.Option(..., "--dir", "-d", help="Directory containing ThermoML XML files.", exists=True, file_okay=False, dir_okay=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save .parsed.pkl files. Defaults to input_dir.", file_okay=False, dir_okay=True, writable=True),
    overwrite: bool = typer.Option(False, "--overwrite", "-ow", help="Overwrite existing .pkl files."),
    max_workers: Optional[int] = typer.Option(None, help="Maximum number of worker processes. Defaults to number of CPUs. (Currently runs sequentially)") # Clarified sequential execution
):
    """
    Parse all ThermoML XML files in a directory.
    Caches results as .pkl files, which are used by `build-dataframe` for faster processing.
    Currently runs sequentially.
    """
    typer.echo(f"Starting to parse all XML files in: {input_dir}")
    
    schema_obj = get_schema_object()
    if schema_obj is None:
        typer.echo("[ERROR] Schema could not be loaded. Cannot parse files.", err=True)
        raise typer.Exit(code=2)

    xml_files = list(input_dir.rglob("*.xml"))
    if not xml_files:
        typer.echo("No XML files found in the directory.")
        return

    typer.echo(f"Found {len(xml_files)} XML files to process.")

    actual_output_dir = output_dir if output_dir else input_dir
    if output_dir: # Ensure output_dir exists if specified
        output_dir.mkdir(parents=True, exist_ok=True)

    parsed_count = 0
    failed_count = 0

    for xml_file in xml_files:
        output_pkl_path = actual_output_dir / xml_file.with_suffix(".parsed.pkl").name
        
        if output_pkl_path.exists() and not overwrite:
            logger.info(f"Skipping {xml_file}, output {output_pkl_path} already exists.")
            # typer.echo(f"Skipping {xml_file}, output {output_pkl_path} already exists.") # Optional: more verbose
            continue

        try:
            logger.info(f"Parsing {xml_file}...")
            records = parse_thermoml_xml(str(xml_file), schema_obj) # Pass schema object
            record_dicts = [_to_dict_for_pickle(r) for r in records]
            with open(output_pkl_path, "wb") as f:
                pickle.dump(record_dicts, f)
            logger.info(f"Successfully parsed {xml_file} -> {output_pkl_path}")
            parsed_count += 1
        except ValueError as e: # Catch schema validation errors or parsing errors
            logger.error(f"Failed to parse {xml_file}: {e}. See debug log for details.")
            # typer.echo(f"[ERROR] Failed to parse {xml_file}: {e}", err=True) # Optional: more verbose
            failed_count += 1
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing {xml_file}: {e}")
            # typer.echo(f"[ERROR] An unexpected error occurred while parsing {xml_file}: {e}", err=True) # Optional: more verbose
            failed_count += 1
    
    typer.echo(f"\\nParsing complete. Successfully parsed: {parsed_count}, Failed: {failed_count}")


@app.command()
def build_dataframe(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory containing ThermoML XML files or .parsed.pkl cache files.", exists=True, file_okay=False, dir_okay=True, readable=True),
    output_data_file: Path = typer.Option("thermoml_data.csv", "--output-data-file", "-od", help="Output file for the main data (e.g., data.csv, data.h5, data.parquet). Format inferred from extension."),
    output_compounds_file: Path = typer.Option("thermoml_compounds.csv", "--output-compounds-file", "-oc", help="Output file for compounds data (e.g., compounds.csv, compounds.h5, compounds.parquet). Format inferred from extension."),
    output_repo_metadata_file: Optional[Path] = typer.Option(None, "--output-repo-metadata-file", "-om", help="Output JSON file for repository metadata."),
    repo_metadata_path: Optional[Path] = typer.Option(None, help="Path to archive_info.json for repository metadata. If not provided, tries input_dir then default location."),
    normalize_alloys_flag: bool = typer.Option(False, "--normalize-alloys", help="Enable alloy normalization (requires pymatgen)."),
    output_hdf_key_data: str = typer.Option("data", help="Key for main data in HDF5 output."),
    output_hdf_key_compounds: str = typer.Option("compounds", help="Key for compounds data in HDF5 output.")
):
    """
    Builds DataFrames from ThermoML XML files.
    If .parsed.pkl cache files (generated by `parse-all`) are present and up-to-date,
    this command will use them to accelerate processing. Otherwise, it will parse the XML files directly.
    Saves the main data, compounds data, and repository metadata.
    """
    typer.echo(f"Building DataFrames from: {input_dir}")

    schema_obj = get_schema_object() # Load schema for potential direct parsing
    if schema_obj is None:
        # This check is crucial because build_pandas_dataframe will pass this schema_obj
        # to parse_thermoml_xml, which expects a valid XMLSchema object, not None,
        # if it needs to parse any XML files (e.g., cache miss).
        typer.echo("[ERROR] Schema could not be loaded. This is required for parsing XML files if cache is not up-to-date. Cannot build dataframe.", err=True)
        raise typer.Exit(code=2)

    xml_files_paths = [str(f) for f in input_dir.rglob("*.xml")]
    if not xml_files_paths:
        typer.echo("No XML files found in the input directory. Will only use .parsed.pkl files if present.")

    # Determine repository metadata path and load it
    actual_repo_metadata: Optional[dict] = None
    temp_repo_metadata_path_str: Optional[str] = None
    if repo_metadata_path:
        if repo_metadata_path.is_file():
            temp_repo_metadata_path_str = str(repo_metadata_path)
        else:
            typer.echo(f"Warning: Provided repository metadata file not found: {repo_metadata_path}")
    elif (input_dir / "archive_info.json").is_file():
        temp_repo_metadata_path_str = str(input_dir / "archive_info.json")
    elif (Path(os.path.expanduser(THERMOML_PATH)) / "archive_info.json").is_file(): # Check default location
         temp_repo_metadata_path_str = str(Path(os.path.expanduser(THERMOML_PATH)) / "archive_info.json")

    if temp_repo_metadata_path_str:
        try:
            with open(temp_repo_metadata_path_str, "r", encoding="utf-8") as f:
                actual_repo_metadata = json.load(f)
            logger.info(f"Loaded repository metadata from: {temp_repo_metadata_path_str}")
        except Exception as e:
            logger.warning(f"Could not load repository metadata from {temp_repo_metadata_path_str}: {e}")
            typer.echo(f"Warning: Could not load repository metadata from {temp_repo_metadata_path_str}: {e}")
    else:
        logger.info("No repository metadata file specified or found. Proceeding without it.")
        typer.echo("Info: No repository metadata file specified or found.")


    try:
        data_bundle = build_pandas_dataframe(
            xml_files=xml_files_paths,
            normalize_alloys=normalize_alloys_flag,
            repository_metadata=actual_repo_metadata, 
            schema_obj_param=schema_obj
        )

        df_data = data_bundle.get('data')
        df_compounds = data_bundle.get('compounds')
        repo_meta_content = data_bundle.get('repository_metadata')

        # Save main data
        if df_data is not None and not df_data.empty:
            output_data_file_format = output_data_file.suffix.lower()
            output_data_file.parent.mkdir(parents=True, exist_ok=True)
            if output_data_file_format == ".csv":
                df_data.to_csv(output_data_file, index=False)
            elif output_data_file_format in [".h5", ".hdf5"]:
                df_data.to_hdf(output_data_file, key=output_hdf_key_data, mode='w')
            elif output_data_file_format == ".parquet":
                try:
                    df_data.to_parquet(output_data_file, index=False)
                except ImportError as e:
                    typer.echo("[ERROR] Parquet output requires 'pyarrow' or 'fastparquet'. Please install one of these packages.", err=True)
                    raise typer.Exit(code=2)
            else:
                typer.echo(f"Warning: Unsupported output format for data file: {output_data_file_format}. Saving as CSV.")
                df_data.to_csv(output_data_file.with_suffix(".csv"), index=False)
            typer.echo(f"Main data saved to: {output_data_file}")
        else:
            typer.echo("No main data to save or data DataFrame is empty.")

        # Save compounds data
        if df_compounds is not None and not df_compounds.empty:
            output_compounds_file_format = output_compounds_file.suffix.lower()
            output_compounds_file.parent.mkdir(parents=True, exist_ok=True)
            if output_compounds_file_format == ".csv":
                df_compounds.to_csv(output_compounds_file, index=False)
            elif output_compounds_file_format in [".h5", ".hdf5"]:
                df_compounds.to_hdf(output_compounds_file, key=output_hdf_key_compounds, mode='w')
            elif output_compounds_file_format == ".parquet":
                try:
                    df_compounds.to_parquet(output_compounds_file, index=False)
                except ImportError as e:
                    typer.echo("[ERROR] Parquet output requires 'pyarrow' or 'fastparquet'. Please install one of these packages.", err=True)
                    raise typer.Exit(code=2)
            else:
                typer.echo(f"Warning: Unsupported output format for compounds file: {output_compounds_file_format}. Saving as CSV.")
                df_compounds.to_csv(output_compounds_file.with_suffix(".csv"), index=False)
            typer.echo(f"Compounds data saved to: {output_compounds_file}")
        else:
            typer.echo("No compounds data to save or compounds DataFrame is empty.")

        # Save repository metadata
        if output_repo_metadata_file and repo_meta_content:
            output_repo_metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_repo_metadata_file, 'w') as f:
                json.dump(repo_meta_content, f, indent=2)
            typer.echo(f"Repository metadata saved to: {output_repo_metadata_file}")
        elif output_repo_metadata_file:
            typer.echo("No repository metadata to save (either not loaded or not requested).")
    except FileNotFoundError as e: # Specifically for schema not found by get_schema_path
        typer.echo(f"[ERROR] Schema file issue: {e}")
        typer.echo(traceback.format_exc())
        raise typer.Exit(code=2)
    except Exception as e:
        typer.echo(f"Error building DataFrame: {e}")
        typer.echo(traceback.format_exc())
        raise typer.Exit(code=2)


@app.command()
def update_archive():
    """Downloads and extracts the latest ThermoML archive from NIST."""
    typer.echo("Starting ThermoML archive update...")
    try:
        update_archive_core()
        typer.echo("ThermoML archive update process completed successfully.")
    except Exception as e:
        typer.echo(f"Error during archive update: {e}")
        typer.echo(traceback.format_exc())
        raise typer.Exit(code=1)

@app.command()
def search_data(
    data_file: Path = typer.Option(..., "--data-file", "-df", help="Path to the data file (CSV, HDF5, Parquet).", exists=True, file_okay=True, dir_okay=False, readable=True),
    component: Optional[List[str]] = typer.Option(None, "--component", "-c", help="Filter by component name(s) (e.g., water ethanol). All must be present."),
    property_col: Optional[str] = typer.Option(None, "--property", "-p", help="Filter by property column name (e.g., 'prop_Viscosity,_Pa*s'). Checks if property has a value."),
    doi: Optional[str] = typer.Option(None, "--doi", help="Filter by DOI (case-insensitive contains)."),
    author: Optional[str] = typer.Option(None, "--author", help="Filter by first author (case-insensitive contains)."),
    journal: Optional[str] = typer.Option(None, "--journal", help="Filter by journal name (case-insensitive contains)."),
    year: Optional[int] = typer.Option(None, "--year", help="Filter by exact publication year."),
    temp_k_gt: Optional[float] = typer.Option(None, "--temp-k-gt", help="Filter by temperature in Kelvin (greater than). Assumes 'var_Temperature,_K' column."),
    temp_k_lt: Optional[float] = typer.Option(None, "--temp-k-lt", help="Filter by temperature in Kelvin (less than). Assumes 'var_Temperature,_K' column."),
    output_file: Optional[Path] = typer.Option(None, "--output-file", "-o", help="Save filtered results to a CSV file."),
    max_results: Optional[int] = typer.Option(None, "--max-results", "-n", help="Limit the number of results displayed/saved."),
    hdf_key: Optional[str] = typer.Option("data", help="Key for HDF5 input file, if applicable.")
):
    """Search and filter data from a previously built DataFrame file."""
    typer.echo(f"Loading data from: {data_file}")
    try:
        df_format = data_file.suffix.lower()
        if df_format == ".csv":
            df = pd.read_csv(data_file)
        elif df_format in [".h5", ".hdf5"]:
            df = pd.read_hdf(data_file, key=hdf_key)
        elif df_format == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            typer.echo(f"Error: Unsupported data file format: {df_format}. Supported: .csv, .h5, .hdf5, .parquet")
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error loading data file: {e}")
        typer.echo(traceback.format_exc())
        raise typer.Exit(code=1)

    typer.echo(f"Initial records: {len(df)}")
    original_df = df.copy() # Keep original for filtering

    # Apply filters
    if component:
        for comp in component:
            # Ensure 'components' column exists and is string type for .str.contains
            if 'components' in df.columns and pd.api.types.is_string_dtype(df['components']):
                 df = df[df['components'].str.contains(comp, case=False, na=False)]
            else:
                typer.echo(f"Warning: 'components' column not found or not string type. Cannot filter by component: {comp}")
                break # Stop component filtering if column is problematic
        typer.echo(f"After component filter ({component}): {len(df)} records")


    if property_col:
        if property_col in df.columns:
            df = df[df[property_col].notna()]
            typer.echo(f"After property filter ({property_col}): {len(df)} records")
        else:
            typer.echo(f"Warning: Property column '{property_col}' not found.")


    if doi:
        if 'doi' in df.columns and pd.api.types.is_string_dtype(df['doi']):
            df = df[df['doi'].str.contains(doi, case=False, na=False)]
            typer.echo(f"After DOI filter ({doi}): {len(df)} records")
        else:
            typer.echo("Warning: 'doi' column not found or not string type. Cannot filter by DOI.")


    if author: # Assuming author is in a column like 'sAuthor(s)' or similar
        # This requires knowing the exact column name for authors.
        # For now, let's assume a generic 'authors' column or skip if not obvious.
        author_col_guess = next((col for col in df.columns if 'author' in col.lower()), None)
        if author_col_guess and pd.api.types.is_string_dtype(df[author_col_guess]):
            df = df[df[author_col_guess].str.contains(author, case=False, na=False)]
            typer.echo(f"After author filter ({author} in {author_col_guess}): {len(df)} records")
        else:
            typer.echo("Warning: Author column not found or not string type. Cannot filter by author.")


    if journal: # Assuming journal is in 'sJournal' or similar
        journal_col_guess = next((col for col in df.columns if 'journal' in col.lower()), None)
        if journal_col_guess and pd.api.types.is_string_dtype(df[journal_col_guess]):
            df = df[df[journal_col_guess].str.contains(journal, case=False, na=False)]
            typer.echo(f"After journal filter ({journal} in {journal_col_guess}): {len(df)} records")
        else:
            typer.echo("Warning: Journal column not found or not string type. Cannot filter by journal.")


    if year:
        if 'publication_year' in df.columns:
            # Ensure the column is numeric before comparison
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
            df = df[df['publication_year'] == year]
            typer.echo(f"After year filter ({year}): {len(df)} records")
        else:
            typer.echo("Warning: 'publication_year' column not found. Cannot filter by year.")

    temp_col_name = 'var_Temperature,_K' # Standardized temperature column name
    if temp_k_gt is not None:
        if temp_col_name in df.columns:
            df[temp_col_name] = pd.to_numeric(df[temp_col_name], errors='coerce')
            df = df[df[temp_col_name] > temp_k_gt]
            typer.echo(f"After temperature > {temp_k_gt}K filter: {len(df)} records")
        else:
            typer.echo(f"Warning: Temperature column '{temp_col_name}' not found.")


    if temp_k_lt is not None:
        if temp_col_name in df.columns:
            df[temp_col_name] = pd.to_numeric(df[temp_col_name], errors='coerce')
            df = df[df[temp_col_name] < temp_k_lt]
            typer.echo(f"After temperature < {temp_k_lt}K filter: {len(df)} records")
        else:
            typer.echo(f"Warning: Temperature column '{temp_col_name}' not found.")


    num_results = len(df)
    typer.echo(f"\nFound {num_results} matching records.")

    results_to_show_df = df.head(max_results) if max_results is not None else df

    if output_file:
        results_to_show_df.to_csv(output_file, index=False)
        typer.echo(f"Results saved to: {output_file}")
    elif not results_to_show_df.empty:
        typer.echo("Displaying results:")
        try:
            from tabulate import tabulate as tabulate_lib # Renamed to avoid conflict
            # Ensure results_to_show_df is a DataFrame for tabulate
            if isinstance(results_to_show_df, pd.Series):
                results_to_show_df_display = results_to_show_df.to_frame().T
            else:
                results_to_show_df_display = results_to_show_df
            
            # Select a subset of columns for display if too many
            display_columns = list(results_to_show_df_display.columns)
            if len(display_columns) > 10:
                display_columns = display_columns[:5] + ['...'] + display_columns[-4:]
                # This is a conceptual slice; tabulate needs actual data
                # For simplicity, we'll just tabulate what we have or let tabulate handle it.

            typer.echo(tabulate_lib(results_to_show_df_display, headers="keys", tablefmt="grid", showindex=False))
        except ImportError:
            typer.echo("Tabulate library not found. Printing basic DataFrame string.")
            typer.echo(results_to_show_df.to_string(index=False))
        except Exception as e:
            typer.echo(f"Error during tabulate display: {e}. Printing basic DataFrame string.")
            typer.echo(results_to_show_df.to_string(index=False))

    elif num_results > 0 and results_to_show_df.empty and max_results == 0:
         typer.echo("Max results set to 0, so no records are displayed or saved.")
    elif num_results == 0:
        typer.echo("No records to display or save.")


@app.command()
def summarize_archive(
    source: Path = typer.Option(..., "--source", "-s", help="Path to data file (CSV, HDF5, Parquet) or directory of XML files.", exists=True, readable=True),
    repo_metadata_path: Optional[Path] = typer.Option(None, "--repo-metadata-path", "-rm", help="Path to archive_info.json for repository metadata (used if source is a directory).")
):
    """Provides a summary of a ThermoML data file or an archive directory."""
    summary = {}
    if source.is_file():
        typer.echo(f"Summarizing data from file: {source}")
        try:
            df_format = source.suffix.lower()
            if df_format == ".csv":
                df = pd.read_csv(source)
            elif df_format in [".h5", ".hdf5"]:
                df = pd.read_hdf(source, key="data") # Assuming default key "data"
            elif df_format == ".parquet":
                df = pd.read_parquet(source)
            else:
                typer.echo(f"Error: Unsupported data file format: {df_format}")
                raise typer.Exit(code=1)

            summary["source_type"] = "DataFrame File"
            summary["file_path"] = str(source)
            summary["total_records"] = len(df)
            if 'components' in df.columns:
                # Assuming 'components' is a string like "water, ethanol"
                all_components = set()
                for comp_list_str in df['components'].dropna().astype(str):
                    all_components.update([c.strip() for c in comp_list_str.split(',')])
                summary["unique_components_count"] = len(all_components)
                summary["unique_components_sample"] = sorted(list(all_components))[:10] # Sample
            else:
                summary["unique_components_count"] = "N/A ('components' column missing)"


            prop_cols = [col for col in df.columns if col.startswith("prop_")]
            summary["property_types_count"] = len(prop_cols)
            summary["property_types_available"] = prop_cols

            if 'publication_year' in df.columns:
                years = pd.to_numeric(df['publication_year'], errors='coerce').dropna()
                summary["min_publication_year"] = int(years.min()) if not years.empty else "N/A"
                summary["max_publication_year"] = int(years.max()) if not years.empty else "N/A"
            else:
                summary["min_publication_year"] = "N/A ('publication_year' column missing)"
                summary["max_publication_year"] = "N/A ('publication_year' column missing)"


            if 'doi' in df.columns:
                summary["unique_dois_count"] = df['doi'].nunique()
            else:
                summary["unique_dois_count"] = "N/A ('doi' column missing)"


            # Print summary as table (existing behavior)
            typer.echo("Summary:")
            for k, v in summary.items():
                typer.echo(f"{k}: {v}")
            # Add explicit summary line for test assertion
            typer.echo(f"total_records: {summary['total_records']}")

        except Exception as e:
            typer.echo(f"Error processing data file {source}: {e}")
            typer.echo(traceback.format_exc())
            raise typer.Exit(code=1)

    elif source.is_dir():
        typer.echo(f"Summarizing XML files from directory: {source}")
        xml_files = list(source.rglob("*.xml"))
        summary["source_type"] = "XML Directory"
        summary["directory_path"] = str(source)
        summary["total_xml_files"] = len(xml_files)
        typer.echo("Summary for XML directory (file count only). For detailed content summary, run `build-dataframe` first and summarize the output file.")
        # Add explicit summary line for test assertion
        typer.echo(f"total_xml_files: {summary['total_xml_files']}")

        actual_repo_metadata_path_to_load: Optional[Path] = None
        if repo_metadata_path and repo_metadata_path.is_file():
            actual_repo_metadata_path_to_load = repo_metadata_path
        elif (source / "archive_info.json").is_file():
            actual_repo_metadata_path_to_load = source / "archive_info.json"
        else:
            default_archive_info = Path(os.path.expanduser(THERMOML_PATH)) / "archive_info.json"
            if default_archive_info.is_file():
                actual_repo_metadata_path_to_load = default_archive_info

        if actual_repo_metadata_path_to_load:
            try:
                import json # Ensure json is imported in this scope if not already
                with open(actual_repo_metadata_path_to_load, 'r', encoding="utf-8") as f:
                    repo_meta = json.load(f)
                summary["repository_title"] = repo_meta.get("title")
                summary["repository_version"] = repo_meta.get("version")
                summary["repository_description"] = repo_meta.get("description")
                summary["repository_retrieved_date"] = repo_meta.get("retrieved_date")
            except Exception as e:
                typer.echo(f"Warning: Could not load or parse repository metadata from {actual_repo_metadata_path_to_load}: {e}")
        else:
            typer.echo("No repository metadata (archive_info.json) found or specified.")


    else:
        typer.echo(f"Error: Source path {source} is not a file or directory.")
        raise typer.Exit(code=1)

    # Print summary using Rich Table
    table = Table(title="Archive Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in summary.items():
        if isinstance(value, list):
            table.add_row(key.replace("_", " ").title(), ", ".join(map(str, value)) if value else "N/A")
        else:
            table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print("\n--- Archive Summary ---")
    console.print(table)
    console.print("-----------------------")


@app.command()
def convert_format(
    input_file: Path = typer.Option(..., "--input-file", "-i", help="Input data file (CSV, HDF5, Parquet).", file_okay=True, dir_okay=False, readable=True),
    output_file: Path = typer.Option(..., "--output-file", "-o", help="Output data file (CSV, HDF5, Parquet)."),
    input_hdf_key: str = typer.Option("data", help="Key for input HDF5 file, if applicable."), # Changed Optional[str] to str
    output_hdf_key: str = typer.Option("data", help="Key for output HDF5 file, if applicable.") # Changed Optional[str] to str
):
    """Converts data files between supported formats (CSV, HDF5, Parquet)."""
    if not input_file.exists():
        msg = f"File not found: {input_file}"
        typer.echo(msg)
        typer.echo(msg, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"Converting {input_file} to {output_file}...")

    input_format = input_file.suffix.lower()
    output_format = output_file.suffix.lower()

    if input_format not in SUPPORTED_DATAFRAME_FORMATS or output_format not in SUPPORTED_DATAFRAME_FORMATS:
        typer.echo(f"Error: Unsupported file format. Supported formats are: {', '.join(SUPPORTED_DATAFRAME_FORMATS)}")
        raise typer.Exit(code=3) # Specific exit code for format error

    try:
        df = None
        if input_format == ".csv":
            df = pd.read_csv(input_file)
        elif input_format in [".h5", ".hdf5"]:
            df = pd.read_hdf(input_file, key=input_hdf_key)
        elif input_format == ".parquet":
            df = pd.read_parquet(input_file)

        if df is None: # Should not happen if format check passed
            typer.echo("Error: Could not load input DataFrame.")
            raise typer.Exit(code=1)

        output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        if output_format == ".csv":
            df.to_csv(output_file, index=False)
        elif output_format in [".h5", ".hdf5"]:
            df.to_hdf(output_file, key=output_hdf_key, mode='w')
        elif output_format == ".parquet":
            df.to_parquet(output_file, index=False)
        
        typer.echo(f"Successfully converted and saved to {output_file}")

    except Exception as e:
        typer.echo(f"Error during conversion: {e}", err=True)
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=1)

@app.command()
def clear_cache(
    directory: Optional[Path] = typer.Option(None, "--dir", "-d", help="Directory to clear .parsed.pkl files from. Defaults to THERMOML_PATH or ~/.thermoml/.", file_okay=False, dir_okay=True, readable=True),
    yes: bool = typer.Option(False, "--yes", "-y", help="Automatically confirm deletion without prompting.")
):
    """
    Deletes all *.parsed.pkl cache files from the specified directory (and subdirectories).
    If no directory is specified, it uses the path from the THERMOML_PATH environment
    variable, or defaults to ~/.thermoml/.
    """
    target_dir: Path
    if directory:
        target_dir = directory
        if not target_dir.is_dir():
            typer.echo(f"Error: Provided directory does not exist: {target_dir}")
            raise typer.Exit(code=1)
    else:
        # Use THERMOML_PATH from config, which already handles env var and default
        target_dir = Path(os.path.expanduser(THERMOML_PATH))
        if not target_dir.exists():
            typer.echo(f"Info: Default cache directory does not exist: {target_dir}. Nothing to clear.")
            raise typer.Exit(code=0)

    typer.echo(f"Searching for .parsed.pkl files in {target_dir} and its subdirectories...")
    cache_files = list(target_dir.rglob("*.parsed.pkl"))

    if not cache_files:
        typer.echo("No .parsed.pkl files found to delete.")
        raise typer.Exit(code=0)

    typer.echo(f"Found {len(cache_files)} .parsed.pkl files:")
    for f_path in cache_files:
        typer.echo(f"  - {f_path}")

    if not yes:
        confirmation = typer.confirm("Are you sure you want to delete these files?")
        if not confirmation:
            typer.echo("Aborted by user.")
            raise typer.Exit(code=0)
    for f_path in cache_files:
        try:
            f_path.unlink()
            typer.echo(f"Successfully deleted: {f_path}")
        except Exception as e:
            typer.echo(f"Failed to delete {f_path}: {e}")
    typer.echo(f"Successfully deleted {len(cache_files)} .parsed.pkl file(s)")
    typer.echo("Cache clearing complete.")

if __name__ == "__main__":
    app()