import pytest
from pathlib import Path
from typer.testing import CliRunner
from thermoml_fair.scripts.cli import app # Corrected import
import tempfile
import os
import shutil
import time # For managing timestamps
import pickle # For inspecting .pkl files
import pandas as pd

runner = CliRunner()

# Existing sample files
SAMPLE_XML_DIR = Path("thermoml_fair/data/test_data") # Corrected path relative to package
SAMPLE_XML_1 = SAMPLE_XML_DIR / "j.tca.2007.01.009.xml"
SAMPLE_XML_2 = SAMPLE_XML_DIR / "acs.jced.8b00050.xml" # Add another for variety
# SCHEMA_PATH is no longer needed here as CLI handles it

# Helper to create a temporary test environment with sample XMLs
@pytest.fixture
def temp_xml_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Copy sample XMLs into the temp directory
        shutil.copy(SAMPLE_XML_1, tmppath / SAMPLE_XML_1.name)
        shutil.copy(SAMPLE_XML_2, tmppath / SAMPLE_XML_2.name)
        yield tmppath

def test_cli_parse(temp_xml_dir):  # Use the existing temp_xml_dir fixture
    # The input XML file is now the copy in the temporary directory
    input_file_in_temp = temp_xml_dir / SAMPLE_XML_1.name

    # The output .pkl file will also be in this temporary directory
    expected_output_pkl_path = input_file_in_temp.with_suffix(".parsed.pkl")

    result = runner.invoke(app, [
        "parse",
        "--file", str(input_file_in_temp),
        # By default, output goes next to the input file.
        # Since input_file_in_temp is in a clean temporary directory,
        # no --overwrite flag is needed.
    ])

    assert result.exit_code == 0, f"CLI command failed with exit code {result.exit_code}. Output:\n{result.stdout}"
    assert expected_output_pkl_path.exists()
    # Verify that the .pkl file contains data (basic check)
    with open(expected_output_pkl_path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, tuple) # Data should be a tuple (records, compounds)
    assert isinstance(data[0], list) # First element (records) should be a list
    assert len(data[0]) > 0  # Assuming the sample XML has records

def test_cli_validate_valid():
    result = runner.invoke(app, [
        "validate",
        "--file", str(SAMPLE_XML_1),
    ])
    assert result.exit_code == 0
    assert "is valid" in result.stdout

def test_cli_validate_missing_file():
    result = runner.invoke(app, ["validate", "--file", "non_existent_file.xml"])
    assert result.exit_code != 0  # Expect non-zero for error
    assert "File not found" in result.stdout # Or similar error message

def test_cli_parse_all_creates_caches(temp_xml_dir):
    """Test that `parse-all` creates .parsed.pkl files."""
    result = runner.invoke(app, [
        "parse-all",
        "--dir", str(temp_xml_dir),
    ], catch_exceptions=False)
    assert result.exit_code == 0
    assert (temp_xml_dir / SAMPLE_XML_1.with_suffix(".parsed.pkl").name).exists()
    assert (temp_xml_dir / SAMPLE_XML_2.with_suffix(".parsed.pkl").name).exists()
    assert "Successfully parsed" in result.stdout # Check for success messages

def test_cli_build_dataframe_uses_fresh_caches(temp_xml_dir):
    """Test `build-dataframe` uses .pkl caches created by `parse-all`."""
    # 1. Run parse-all
    parse_result = runner.invoke(app, [
        "parse-all", "--dir", str(temp_xml_dir),
    ], catch_exceptions=False)
    assert parse_result.exit_code == 0
    pkl_file_1 = temp_xml_dir / SAMPLE_XML_1.with_suffix(".parsed.pkl").name
    assert pkl_file_1.exists()

    # Ensure pkl file is fresh (optional: make it slightly newer for robustness if needed)
    # For this test, we assume parse-all makes it fresh enough.

    # 2. Run build-dataframe
    output_data_csv = temp_xml_dir / "data.csv"
    output_compounds_csv = temp_xml_dir / "compounds.csv"
    build_result = runner.invoke(app, [
        "build-dataframe",
        "--input-dir", str(temp_xml_dir),
        "--output-data-file", str(output_data_csv),
        "--output-compounds-file", str(output_compounds_csv),
    ], catch_exceptions=False)
    assert build_result.exit_code == 0
    assert output_data_csv.exists()
    # A simple check that the CSV is not empty
    df = pd.read_csv(output_data_csv)
    assert not df.empty
    # Ideally, check for log message indicating cache was used, if utils.py logs this clearly.
    # For now, successful creation implies it worked.

def test_cli_build_dataframe_parses_xml_if_no_cache(temp_xml_dir):
    """Test `build-dataframe` parses XML directly if no .pkl cache exists."""
    # Ensure no .pkl files exist
    for pkl_file in temp_xml_dir.glob("*.parsed.pkl"):
        pkl_file.unlink()

    output_data_csv = temp_xml_dir / "data_no_cache.csv"
    result = runner.invoke(app, [
        "build-dataframe",
        "--input-dir", str(temp_xml_dir),
        "--output-data-file", str(output_data_csv),
    ])
    assert result.exit_code == 0
    assert output_data_csv.exists()
    df = pd.read_csv(output_data_csv)
    assert not df.empty
    # Check for log message indicating XML was parsed (if available and distinct)

def test_cli_build_dataframe_parses_xml_if_cache_is_stale(temp_xml_dir):
    """Test `build-dataframe` re-parses XML if .pkl cache is stale."""
    # 1. Run parse-all to create initial cache
    runner.invoke(app, ["parse-all", "--dir", str(temp_xml_dir)])
    pkl_file_1 = temp_xml_dir / SAMPLE_XML_1.with_suffix(".parsed.pkl").name
    xml_file_1 = temp_xml_dir / SAMPLE_XML_1.name
    assert pkl_file_1.exists()

    # 2. Make XML file newer than PKL file
    time.sleep(0.1) # Ensure modification time is different
    xml_file_1.touch() # Update XML modification time

    # Store original content of pkl to compare later (simplified check)
    # A more robust check would involve parsing the XML and comparing data content
    # For this test, we'll rely on the logic in build_pandas_dataframe to re-parse.

    output_data_csv = temp_xml_dir / "data_stale_cache.csv"
    result = runner.invoke(app, [
        "build-dataframe",
        "--input-dir", str(temp_xml_dir),
        "--output-data-file", str(output_data_csv),
    ])
    assert result.exit_code == 0
    assert output_data_csv.exists()
    # We expect it to re-parse. A more advanced test would modify XML content
    # and check if the DataFrame reflects the modification.
    # For now, we assume the timestamp check in build_pandas_dataframe works.

def test_cli_parse_all_overwrite(temp_xml_dir):
    """Test `parse-all --overwrite` re-parses files."""
    # 1. Initial parse
    runner.invoke(app, ["parse-all", "--dir", str(temp_xml_dir)])
    pkl_file_1 = temp_xml_dir / SAMPLE_XML_1.with_suffix(".parsed.pkl").name
    initial_mtime = pkl_file_1.stat().st_mtime

    # 2. Wait a bit, then parse with overwrite
    time.sleep(0.1)
    result = runner.invoke(app, [
        "parse-all", "--dir", str(temp_xml_dir), "--overwrite"
    ])
    assert result.exit_code == 0
    assert pkl_file_1.stat().st_mtime > initial_mtime # PKL should be newer

# --- Basic tests for other new commands ---

def test_cli_build_dataframe_output_formats(temp_xml_dir):
    """Test different output formats for build-dataframe."""
    formats_to_test = {
        "csv": (temp_xml_dir / "data.csv", temp_xml_dir / "compounds.csv"),
        "h5": (temp_xml_dir / "data.h5", temp_xml_dir / "compounds.h5"),
        "parquet": (temp_xml_dir / "data.parquet", temp_xml_dir / "compounds.parquet"),
    }
    # Check for Parquet support
    parquet_supported = True
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            parquet_supported = False
    for fmt, (data_file, compounds_file) in formats_to_test.items():
        if fmt == "parquet" and not parquet_supported:
            pytest.skip("Skipping Parquet test: pyarrow or fastparquet not installed.")
        result = runner.invoke(app, [
            "build-dataframe",
            "--input-dir", str(temp_xml_dir),
            "--output-data-file", str(data_file),
            "--output-compounds-file", str(compounds_file),
        ])
        assert result.exit_code == 0, f"Failed for format {fmt}: {result.stdout}"
        assert data_file.exists()
        # compounds_file might be empty if no compound data, so just check existence
        assert compounds_file.exists()
        # Basic load check
        if fmt == "csv": pd.read_csv(data_file)
        elif fmt == "h5": pd.read_hdf(data_file, key="data") # Assuming default key "data"
        elif fmt == "parquet": pd.read_parquet(data_file)

def test_cli_update_archive(mocker):
    """Test `update-archive` calls the core function."""
    mock_update_core = mocker.patch("thermoml_fair.scripts.cli.update_archive_core")
    result = runner.invoke(app, ["update-archive"])
    assert result.exit_code == 0
    mock_update_core.assert_called_once()
    assert "Update archive process finished." in result.stdout

def test_cli_search_data(temp_xml_dir):
    """Basic test for `search-data`."""
    # 1. Create a sample data file
    data_csv = temp_xml_dir / "search_data_input.csv"
    df_content = pd.DataFrame({
        "components": ["water, ethanol", "water, methanol", "toluene"],
        "prop_Density,_kg/m3": [998.0, 950.0, 867.0],
        "doi": ["10.1000/jced.1", "10.1000/jced.2", "10.1000/jced.3"],
        "publication_year": [2020, 2021, 2020]
    })
    df_content.to_csv(data_csv, index=False)

    # 2. Run search
    result = runner.invoke(app, [
        "search-data",
        "--data-file", str(data_csv),
        "--component", "water",
        "--year", "2020"
    ])
    assert result.exit_code == 0
    assert "Found 1 matching records." in result.stdout # water AND 2020
    assert "10.1000/jced.1" in result.stdout # Check if the correct DOI is in output

def test_cli_summarize_archive_dataframe(temp_xml_dir):
    """Basic test for `summarize-archive` with a DataFrame."""
    data_csv = temp_xml_dir / "summary_data_input.csv"
    df_content = pd.DataFrame({"components": ["water, ethanol"], "doi": ["10.1000/jced.1"], "publication_year": [2020]})
    df_content.to_csv(data_csv, index=False)

    result = runner.invoke(app, ["summarize-archive", "--source", str(data_csv)])
    assert result.exit_code == 0
    assert "Archive Summary" in result.stdout
    assert "total_records: 1" in result.stdout

def test_cli_summarize_archive_xml_dir(temp_xml_dir):
    """Basic test for `summarize-archive` with an XML directory."""
    result = runner.invoke(app, ["summarize-archive", "--source", str(temp_xml_dir)])
    assert result.exit_code == 0
    assert "Archive Summary" in result.stdout
    assert f"total_xml_files: 2" in result.stdout # We copied 2 XMLs

def test_cli_convert_format(temp_xml_dir):
    """Basic test for `convert-format`."""
    # Check for Parquet support
    parquet_supported = True
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            parquet_supported = False
    if not parquet_supported:
        pytest.skip("Skipping Parquet convert-format test: pyarrow or fastparquet not installed.")
    input_csv = temp_xml_dir / "convert_input.csv"
    output_parquet = temp_xml_dir / "convert_output.parquet"
    df_content = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_content.to_csv(input_csv, index=False)

    result = runner.invoke(app, [
        "convert-format",
        "--input-file", str(input_csv),
        "--output-file", str(output_parquet)
    ])
    assert result.exit_code == 0
    assert output_parquet.exists()
    df_parquet = pd.read_parquet(output_parquet)
    pd.testing.assert_frame_equal(df_content, df_parquet)

def test_cli_clear_cache(temp_xml_dir):
    """Test the clear-cache command."""
    # 1. Create some dummy .parsed.pkl files
    (temp_xml_dir / "file1.parsed.pkl").touch()
    (temp_xml_dir / "subdir").mkdir()
    (temp_xml_dir / "subdir" / "file2.parsed.pkl").touch()

    # 2. Run clear-cache
    result = runner.invoke(app, ["clear-cache", "--dir", str(temp_xml_dir), "--yes"])
    assert result.exit_code == 0
    assert "Successfully deleted" in result.stdout
    assert not (temp_xml_dir / "file1.parsed.pkl").exists()
    assert not (temp_xml_dir / "subdir" / "file2.parsed.pkl").exists()

def test_cli_clear_cache_no_files(temp_xml_dir):
    """Test clear-cache when no .parsed.pkl files are present."""
    result = runner.invoke(app, ["clear-cache", "--dir", str(temp_xml_dir), "--yes"])
    assert result.exit_code == 0
    assert "No .parsed.pkl files found to delete" in result.stdout

# --- Tests for data exploration commands ---

@pytest.fixture
def properties_file(temp_xml_dir):
    """Fixture to create a sample properties file for testing."""
    # Run build-dataframe to generate the properties file
    properties_csv = temp_xml_dir / "test_properties.csv"
    data_csv = temp_xml_dir / "test_data.csv"
    compounds_csv = temp_xml_dir / "test_compounds.csv"  # Will be created anyway

    result = runner.invoke(app, [
        "build-dataframe",
        "--input-dir", str(temp_xml_dir),
        "--output-data-file", str(data_csv),
        "--output-compounds-file", str(compounds_csv),
        "--output-properties-file", str(properties_csv),
        "--max-workers", "1",
    ], catch_exceptions=False)

    assert result.exit_code == 0, f"build-dataframe for properties failed: {result.stdout}"
    assert properties_csv.exists(), "Properties file was not created."
    yield properties_csv


def test_cli_properties(properties_file):
    """Test the `properties` command to ensure it lists unique property names from a CSV file."""
    result = runner.invoke(app, [
        "properties",
        "--properties-file", str(properties_file)
    ])
    assert result.exit_code == 0
    assert "Unique property names found" in result.stdout
    # Check for known properties from the sample XML files
    assert "Thermal conductivity, W/m/K" in result.stdout
    #assert "Pressure, kPa" in result.stdout

@pytest.fixture
def compounds_file(temp_xml_dir):
    """Fixture to create a sample compounds file for testing."""
    # Run build-dataframe to generate the compounds file
    compounds_csv = temp_xml_dir / "test_compounds.csv"
    data_csv = temp_xml_dir / "test_data.csv"
    
    # Use catch_exceptions=False to get a full traceback on error
    result = runner.invoke(app, [
        "build-dataframe",
        "--input-dir", str(temp_xml_dir),
        "--output-data-file", str(data_csv),
        "--output-compounds-file", str(compounds_csv),
        "--max-workers", "1",
    ], catch_exceptions=False)

    # The test will fail here if the command fails, and pytest will show the traceback.
    assert result.exit_code == 0, "build-dataframe failed. See traceback above."
    assert compounds_csv.exists(), "Compounds file was not created."
    yield compounds_csv

def test_cli_chemicals(compounds_file):
    """Test the `chemicals` command with different fields."""
    # Test with default field (sCommonName)
    result_common_name = runner.invoke(app, [
        "chemicals",
        "--compounds-file", str(compounds_file),
    ], catch_exceptions=False)
    assert result_common_name.exit_code == 0
    assert "Unique values for 'sCommonName'" in result_common_name.stdout
    assert "carbon dioxide" in result_common_name.stdout
    assert "tin" in result_common_name.stdout # A chemical from the test data

    # Test with a different field (e.g., sFormulaMolec)
    result_formula = runner.invoke(app, [
        "chemicals",
        "--compounds-file", str(compounds_file),
        "--field", "sFormulaMolec",
    ], catch_exceptions=False)
    assert result_formula.exit_code == 0
    assert "Unique values for 'sFormulaMolec'" in result_formula.stdout
    assert "CO2" in result_formula.stdout
    assert "Sn" in result_formula.stdout # Formula for tin

    # Test with normalized_formula for alloys (if any in test data)
    df = pd.read_csv(compounds_file)
    if "normalized_formula" in df.columns:
        result_normalized = runner.invoke(app, [
            "chemicals",
            "--compounds-file", str(compounds_file),
            "--field", "normalized_formula",
        ], catch_exceptions=False)
        assert result_normalized.exit_code == 0
        assert "Unique values for 'normalized_formula'" in result_normalized.stdout