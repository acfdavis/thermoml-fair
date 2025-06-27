# `thermoml-fair` User Guide

This guide provides a comprehensive overview of the `thermoml-fair` package, covering both command-line and Python API usage. This package is designed to process ThermoML XML files into analysis-ready formats like CSV and Parquet, ensuring data is FAIR (Findable, Accessible, Interoperable, and Reusable).

## Table of Contents

1.  [Installation](#1-installation)
2.  [Command-Line Interface (CLI) Usage](#2-command-line-interface-cli-usage)
    *   [Quickstart](#quickstart)
    *   [Global Options](#global-options)
    *   [Commands](#commands)
        *   [`update-archive`](#update-archive)
        *   [`parse-all`](#parse-all)
        *   [`build-dataframe`](#build-dataframe)
        *   [`search-data`](#search-data)
        *   [`summarize-archive`](#summarize-archive)
        *   [`convert-format`](#convert-format)
        *   [`parse`](#parse)
        *   [`validate`](#validate)
        *   [`clear-cache`](#clear-cache)
        *   [`properties`](#properties)
        *   [`chemicals`](#chemicals)
3.  [Python API Usage](#3-python-api-usage)
    *   [Core Function: `build_pandas_dataframe`](#core-function-build_pandas_dataframe)
    *   [Example Workflow in a Notebook](#example-workflow-in-a-notebook)
4.  [Example: End-to-End Workflow](#4-example-end-to-end-workflow)

---

## 1. Installation

First, ensure you have Python installed. Then, install `thermoml-fair` from PyPI:

```bash
pip install thermoml-fair
```

For features involving alloy normalization, `pymatgen` is required:

```bash
pip install pymatgen
```

For saving dataframes in `.parquet` format, `pyarrow` is required:

```bash
pip install pyarrow
```

---

## 2. Command-Line Interface (CLI) Usage

The `thermoml-fair` CLI provides a powerful set of tools to download, process, and manage ThermoML data archives.

### Quickstart

For a fresh start, run these commands in order:

1.  **Download and extract the data:** This fetches the latest ThermoML archive and unpacks the XML files into the default local directory (`~/.thermoml/extracted`).

    ```bash
    thermoml-fair update-archive
    ```

2.  **Parse all XML files:** This pre-processes all XML files and creates cached `.parsed.pkl` files for much faster DataFrame construction. This step is optional but highly recommended.

    ```bash
    thermoml-fair parse-all
    ```

3.  **Build the analysis-ready DataFrames:** This command reads the raw XMLs (or the cached `.pkl` files if they exist) and generates the final CSV files.

    ```bash
    thermoml-fair build-dataframe
    ```

    This will create `thermoml_data.csv`, `thermoml_compounds.csv`, and `thermoml_properties.csv` in your current directory.

### Global Options

*   `--version`: Show the installed version of `thermoml-fair` and exit.
*   `--help`: Show help for any command.

### Commands

---

#### `update-archive`

Downloads the latest ThermoML archive from the NIST repository, extracts all XML files, and downloads the schema.

**Usage:**

```bash
thermoml-fair update-archive [OPTIONS]
```

**Options:**

*   `--path, -p`: Override the default `THEROML_PATH` (`~/.thermoml`) for this run.
*   `--force-download, -f`: Force re-downloading and extraction even if the archive appears up-to-date.

---

#### `parse-all`

Parses all `.xml` files in a directory and creates `.parsed.pkl` cache files. This significantly speeds up subsequent runs of `build-dataframe`.

**Usage:**

```bash
thermoml-fair parse-all [OPTIONS]
```

**Options:**

*   `--dir, -d`: The directory containing the XML files. Defaults to the directory populated by `update-archive`.
*   `--output-dir, -o`: Where to save the `.parsed.pkl` files. Defaults to the input directory.
*   `--overwrite, -ow`: Overwrite existing `.parsed.pkl` files.
*   `--max-workers, -mw`: Number of CPU cores to use for parallel processing. Defaults to all available cores.

---

#### `build-dataframe`

The main command to generate analysis-ready data files from the ThermoML XMLs. It intelligently uses cached `.parsed.pkl` files if they exist.

**Usage:**

```bash
thermoml-fair build-dataframe [OPTIONS]
```

**Key Options:**

*   `--input-dir, -i`: Directory containing the XML and/or `.parsed.pkl` files. Defaults to the one used by `update-archive` and `parse-all`.
*   `--output-data-file, -od`: Path for the main data file. Format (`.csv`, `.parquet`, `.h5`) is inferred from the extension. Default: `thermoml_data.csv`.
*   `--output-compounds-file, -oc`: Path for the compounds data file. Default: `thermoml_compounds.csv`.
*   `--output-properties-file, -op`: Path for the unique properties list. Default: `thermoml_properties.csv`.
*   `--output-repo-metadata-file, -om`: Output JSON file for repository metadata.
*   `--repo-metadata-path, -rmp`: Path to `archive_info.json` for repository metadata.
*   `--normalize-alloys`: (Flag) Enable normalization of alloy compositions and generate a `formula` column for use with `matminer`. Requires `pymatgen`.
*   `--show-failed-files`: (Flag) If any XML files fail to parse, this will print a detailed list of them. The process will **not** halt on failed files.
*   `--max-workers, -mw`: Number of CPU cores to use for parallel processing.

---

#### `search-data`

Search and filter data from a previously built DataFrame file.

**Usage:**

```bash
thermoml-fair search-data --data-file /path/to/data.csv [OPTIONS]
```

**Options:**

*   `--data-file, -df`: (Required) Path to the data file (CSV, HDF5, Parquet).
*   `--component, -c`: Filter by one or more component names (e.g., 'water').
*   `--property, -p`: Filter by a specific property column name.
*   `--doi`: Filter by publication DOI.
*   `--author`: Filter by the first author's name.
*   `--journal`: Filter by the journal name.
*   `--year`: Filter by publication year.
*   `--temp-k-gt`: Filter by temperature in Kelvin (greater than).
*   `--temp-k-lt`: Filter by temperature in Kelvin (less than).
*   `--output-file, -o`: Save the filtered results to a new CSV file.
*   `--max-results, -n`: Limit the number of results returned.
*   `--hdf-key`: Key for HDF5 input file, if applicable.

---

#### `summarize-archive`

Provides a summary of a ThermoML data file or an archive directory.

**Usage:**

```bash
thermoml-fair summarize-archive --source /path/to/data.csv [OPTIONS]
```

**Options:**

*   `--source, -s`: (Required) Path to data file (CSV, HDF5, Parquet) or directory of XML files.
*   `--repo-metadata-path, -rm`: Path to `archive_info.json` for repository metadata.
*   `--hdf-key`: Key for HDF5 input file, if applicable.

---

#### `convert-format`

Converts data files between supported formats (CSV, HDF5, Parquet).

**Usage:**

```bash
thermoml-fair convert-format --input-file /path/to/data.csv --output-file /path/to/data.parquet
```

**Options:**

*   `--input-file, -i`: (Required) Input data file.
*   `--output-file, -o`: (Required) Output data file.
*   `--input-hdf-key`: Key for input HDF5 file.
*   `--output-hdf-key`: Key for output HDF5 file.

---

#### `parse`

Parses a *single* ThermoML XML file and saves the output to a `.parsed.pkl` file. Useful for debugging.

**Usage:**

```bash
thermoml-fair parse --file /path/to/your/file.xml
```

---

#### `validate`

Validates a *single* ThermoML XML file against the official ThermoML schema.

**Usage:**

```bash
thermoml-fair validate --file /path/to/your/file.xml
```

---

#### `clear-cache`

Deletes all `.parsed.pkl` cache files from the specified directory.

**Usage:**

```bash
thermoml-fair clear-cache [OPTIONS]
```

**Options:**

*   `--dir, -d`: Directory to clear `.parsed.pkl` files from. Defaults to the ThermoML data path.
*   `--yes, -y`: Bypass confirmation prompt.

---

#### `properties`

Lists all unique property names from a properties file.

**Usage:**

```bash
thermoml-fair properties --properties-file /path/to/properties.csv
```

**Options:**

*   `--properties-file, -pf`: (Required) Path to the properties CSV file.

---

#### `chemicals`

Lists unique values for a specified field from a compounds data file.

**Usage:**

```bash
thermoml-fair chemicals --compounds-file /path/to/compounds.csv [OPTIONS]
```

**Options:**

*   `--compounds-file`: (Required) Path to the compounds data file.
*   `--field, -f`: Field (column) to display unique values for.
*   `--hdf-key`: Key for HDF5 input file.

---

## 3. Python API Usage

For more programmatic control, you can use the core functions of `thermoml-fair` directly in your Python scripts or Jupyter notebooks.

### Core Function: `build_pandas_dataframe`

This is the central function for data processing. It takes a list of XML file paths and returns a dictionary of pandas DataFrames.

```python
from thermoml_fair.core.utils import build_pandas_dataframe
from thermoml_fair.core.config import THERMOML_SCHEMA_PATH
import xmlschema

# 1. Get the list of your XML files
# (For example, by scanning a directory)
import glob
xml_files = glob.glob('/path/to/your/xmls/*.xml')

# 2. Load the schema (optional but recommended for validation)
schema = xmlschema.XMLSchema(THERMOML_SCHEMA_PATH)

# 3. Call the function
data_bundle = build_pandas_dataframe(
    xml_files=xml_files,
    xsd_path_or_obj=schema,
    normalize_alloys=True,  # Example: enable alloy normalization
    max_workers=4           # Example: use 4 CPU cores
)

# 4. Access your data
df_data = data_bundle['data']
df_compounds = data_bundle['compounds']
df_properties = data_bundle['properties']
failed_files = data_bundle['failed_files']

print(f"Successfully created DataFrame with {len(df_data)} rows.")
if failed_files:
    print(f"Warning: {len(failed_files)} files failed to parse.")

print(df_data.head())
```

### Example Workflow in a Notebook

The `thermoml_data_analysis.ipynb` notebook provided in the repository demonstrates a full workflow, including:

1.  Running the CLI commands directly from the notebook.
2.  Using the Python API to generate the DataFrames.
3.  Performing data cleaning and analysis.
4.  Visualizing the data with Plotly.

---

## 4. Example: End-to-End Workflow

This example demonstrates a complete workflow from downloading data to generating final CSV files, suitable for both a script or a notebook.

```python
import os
from pathlib import Path
import pandas as pd
from thermoml_fair.scripts import cli as thermoml_cli
from thermoml_fair.core.utils import build_pandas_dataframe
from thermoml_fair.core.config import THERMOML_SCHEMA_PATH
import xmlschema
import typer
from typer.testing import CliRunner

# Use the CliRunner for a clean way to invoke the CLI commands
runner = CliRunner()

# --- Step 1: Download and Unpack Data ---
print("Step 1: Updating archive...")
result = runner.invoke(thermoml_cli.app, ["update-archive"])
if result.exit_code != 0:
    raise RuntimeError("Failed to update archive!")
print("Archive updated successfully.")


# --- Step 2: Build DataFrames using the Python API ---
print("\nStep 2: Building DataFrames with the Python API...")

# Get the default path where XMLs were extracted
default_xml_dir = Path(os.path.expanduser("~/.thermoml/extracted"))
xml_files = list(default_xml_dir.glob("*.xml"))

if not xml_files:
    raise FileNotFoundError(f"No XML files found in {default_xml_dir}")

print(f"Found {len(xml_files)} XML files to process.")

# Load schema for validation during parsing
schema = xmlschema.XMLSchema(THERMOML_SCHEMA_PATH)

# Build the dataframes
data_bundle = build_pandas_dataframe(
    xml_files=[str(f) for f in xml_files],
    xsd_path_or_obj=schema,
    normalize_alloys=True,
)

# --- Step 3: Save the DataFrames to CSV ---
print("\nStep 3: Saving DataFrames to CSV files...")

output_dir = Path("./thermoml_output")
output_dir.mkdir(exist_ok=True)

df_data = data_bundle['data']
df_compounds = data_bundle['compounds']
df_properties = data_bundle['properties']

data_path = output_dir / "my_thermoml_data.csv"
compounds_path = output_dir / "my_thermoml_compounds.csv"
properties_path = output_dir / "my_thermoml_properties.csv"

df_data.to_csv(data_path, index=False)
df_compounds.to_csv(compounds_path, index=False)
df_properties.to_csv(properties_path, index=False)

print(f"Data saved to: {data_path}")
print(f"Compounds saved to: {compounds_path}")
print(f"Properties saved to: {properties_path}")

# --- Step 4: Report on Failed Files ---
failed_files = data_bundle['failed_files']
if failed_files:
    print(f"\nWarning: {len(failed_files)} file(s) failed to parse:")
    for f in failed_files:
        print(f"  - {f}")

print("\nWorkflow complete.")
```
