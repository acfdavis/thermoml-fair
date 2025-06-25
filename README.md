# ThermoML-FAIR

ThermoML-FAIR is a modern Python toolkit for downloading, validating, and structuring [ThermoML](https://www.nist.gov/srd/nist-standard-reference-database-229) data from NIST’s ThermoML Archive. Designed for seamless integration with data science and machine learning workflows in materials science, ThermoML-FAIR enables reproducible, automated extraction of thermophysical property data into long-format `pandas` DataFrames, with detailed phase and method information for every measurement.

ThermoML-FAIR is built to support FAIR data practices—making ThermoML data Findable, Accessible, Interoperable, and Reusable. This ensures that data workflows are robust, transparent, and ready for open science and sustainable materials discovery.

This project is a ground-up reimplementation inspired by the original [choderalab/thermopyl](https://github.com/choderalab/thermopyl), rewritten for robust schema validation, high-throughput data processing, and downstream compatibility with tools like Matminer and Citrine. The toolkit is built with sustainability and open science in mind, making it easy to access, analyze, and share high-quality thermophysical property data for materials discovery and informatics.

---

## Features

- **FAIR data principles**: All workflows are designed to make data Findable, Accessible, Interoperable, and Reusable
- **Automated mirroring of the NIST ThermoML Archive** (RSS and archive-based)
- **Schema validation**: All XML files are validated against the official ThermoML XSD
- **Efficient, parallelized parsing and DataFrame construction**: Cross-platform support with `ProcessPoolExecutor` for high-throughput workflows
- **Rich CLI experience**: Intuitive command-line interface with progress bars, robust error handling, and flexible options for parallelism (`--max-workers`)
- **Long-format DataFrame output**: Each measurement is a row, with `phase` and `method` columns included for every property
- **Comprehensive compounds DataFrame**: Always includes a `symbol` column (chemical formula or fallback name) for all files
- **Flexible output**: Export to CSV, HDF5, or Parquet for scalable analytics and ML workflows
- **Resilient download logic**: DOI resolution, override support, and robust error handling
- **Modular, extensible architecture**: Built with `dataclasses`, `pathlib`, and modern Python best practices
- **Ready for ML pipelines**: Designed for easy integration with scikit-learn, matminer, and other data science tools
- **Sustainability focus**: Streamlines reproducible data extraction for green chemistry, energy materials, and more
- **Cache management**: Tools for clearing and managing parsed data caches
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux
- **SPDX License: GPL-2.0**

---

## Installation

ThermoML-FAIR requires Python 3.8 or newer.

**Basic install:**

```bash
pip install thermoml-fair
```

**With Parquet support (recommended for large datasets):**

```bash
pip install 'thermoml-fair[parquet]'
```

### From PyPI (Recommended)

```bash
pip install thermoml-fair
```

### From Source

```bash
git clone https://github.com/your-username/thermoml_fair.git
cd thermoml_fair
pip install .
```

## CLI Usage

The `thermoml_fair` package provides a command-line interface (CLI) for interacting with the ThermoML data archive. All major operations support parallel processing via the `--max-workers` (or `-mw`) option for efficient, scalable workflows.

You can explore the available commands and their options by running:

```bash
thermoml_fair --help
```

And for specific commands:

```bash
thermoml_fair <command> --help
```

### Common Workflow: Step-by-Step

Here's a typical workflow for using the `thermoml_fair` CLI. By default, files are stored in `~/.thermoml/`.

1. **Update the local ThermoML archive:**
   This command downloads or updates the ThermoML XML files from the NIST repository into the default archive directory (`~/.thermoml/archive/`).

   ```bash
   thermoml_fair update-archive
   ```

   To force a re-download of all files:

   ```bash
   thermoml_fair update-archive --force-download
   ```

2. **Parse all downloaded XML files (with parallelism):**
   This command processes all `.xml` files in the archive directory and saves the parsed data as `.parsed.pkl` cache files alongside them.

   ```bash
   thermoml_fair parse-all --max-workers 4
   ```

   You can specify a different directory:

   ```bash
   thermoml_fair parse-all --dir path/to/your/archive --max-workers 2
   ```

3. **Build the consolidated DataFrames (with parallelism):**
   This command consolidates the parsed data into three separate CSV files: one for measurements (`data.csv`), one for compounds (`compounds.csv`), and one for unique property names (`properties.csv`).

   ```bash
   thermoml_fair build-dataframe --max-workers 4
   ```

   You can specify custom output paths and formats (e.g., `parquet`, `h5`):

   ```bash
   thermoml_fair build-dataframe \
     --output-data-file my_data.parquet \
     --output-compounds-file my_compounds.csv \
     --output-properties-file my_properties.csv \
     --max-workers 4
   ```

4. **Explore the Data:**
   After building your data files, you can quickly explore their contents.

   - **List Unique Properties:**

     ```bash
     thermoml_fair properties --properties-file my_properties.csv
     ```

   - **List Unique Chemicals:**

     ```bash
     # List unique common names (default)
     thermoml_fair chemicals --compounds-file my_compounds.csv

     # List unique molecular formulas
     thermoml_fair chemicals --compounds-file my_compounds.csv --field sFormulaMolec
     ```

5. **(Optional) Clear cached files:**
   If you need to clear the `.parsed.pkl` files from the archive directory:

   ```bash
   thermoml_fair clear-cache --yes
   ```

   You can also specify a different directory:

   ```bash
   thermoml_fair clear-cache --dir path/to/your/custom_directory --yes
   ```

**Note:** For every command, you can use the `--help` flag to see all available options and their descriptions. For example, `thermoml_fair update-archive --help`.

## Output Details

- **DataFrame**: Always in long format, with each row representing a single measurement. Includes `phase`, `method`, `property`, `value`, `material_id`, and more.
- **Compounds DataFrame**: Always includes a `symbol` column (chemical formula or fallback name) for all files, supporting robust downstream analytics.
- **Rich CLI output**: Progress bars and status messages for all major operations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For Parquet support, install with `pip install 'thermoml-fair[parquet]'` before running related tests.

---

## Development

To set up a development environment:

```bash
git clone https://github.com/YOURNAME/thermoml-fair.git
cd thermoml-fair
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev,parquet]  # Editable install with dev and Parquet dependencies

# Run tests
pytest
```

Linting and type checking are recommended (e.g., Black, Flake8, MyPy) before committing. ThermoML-FAIR is designed for robust, reproducible data extraction and analysis—ideal for accelerating sustainable materials discovery and informatics workflows.

### Publishing to PyPI

1. **Set up your environment** (first time only):
   - Create a virtual environment:

     ```bash
     python -m venv venv
     ```

   - Activate it. On Windows (PowerShell), you may need to adjust your execution policy:

     ```powershell
     # Allow script execution for the current process, then activate
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
     .\venv\Scripts\Activate.ps1
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - Install build tools:

     ```bash
     pip install build twine
     ```

2. **Build the package**:
   This command creates the `dist/` directory with the source archive and wheel.

   ```bash
   python -m build
   ```

3. **Upload to TestPyPI** (recommended):
   First, make sure you have a TestPyPI account and have configured your `~/.pypirc` file.

   ```bash
   twine upload --repository testpypi dist/*
   ```

4. **Upload to PyPI**:
   Once you've verified the package on TestPyPI, upload it to the official Python Package Index.

   ```bash
   twine upload dist/*
   ```

---

## Project Motivation & Impact

Modern materials science and data-driven discovery require robust, high-quality datasets. ThermoML-FAIR enables rapid, reproducible access to thermophysical property data from the NIST ThermoML Archive, supporting open science, sustainability, and advanced analytics. The toolkit empowers users to build ML-ready datasets for property prediction, process modeling, and informatics, and integrates seamlessly with tools like Matminer for feature extraction and ML pipeline development.

ThermoML-FAIR is a step toward a future where data and automation drive sustainable materials innovation.

---

## About the Author

**Angela C. Davis**  
Materials Scientist passionate about sustainable data-driven discovery and implementation.

- Background: Coatings, thermoplastics, polymer AM, composites, green chemistry, corrosion, advanced manufacturing
- Focus: AI, data science, materials informatics, process modeling, and continuous learning
- Mission: To build tools and workflows that accelerate sustainable innovation and challenge myself to create a better future

**Contact:** <angela.cf.davis@gmail.com>
