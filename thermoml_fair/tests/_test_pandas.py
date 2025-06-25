import os
import shutil
import tempfile
from thermoml_fair.core.utils import get_fn, build_pandas_dataframe, pandas_dataframe

def test_build_pandas_dataframe():
    tmpdir = tempfile.mkdtemp()
    try:
        filenames = [get_fn("je8006138.xml")]
        result = build_pandas_dataframe(filenames)
        data = result["data"]
        compounds = result["compounds"]
        repository_metadata = result["repository_metadata"]

        # Debugging output to understand what was parsed
        print("\n=== DEBUG OUTPUT ===")
        print(f"Parsed {len(data)} records")
        print(f"DataFrame columns: {list(data.columns)}")
        print(f"Compound entries: {len(compounds)}")
        print(f"Repository metadata: {repository_metadata}")
        print(f"First few entries:\n{data.head()}")
        print("=== END DEBUG OUTPUT ===\n")

        # Write and read data to confirm persistence and structure
        data.to_hdf(os.path.join(tmpdir, 'data.h5'), key='data')
        compounds.to_hdf(os.path.join(tmpdir, 'compound_name_to_formula.h5'), key='data') 
        df = pandas_dataframe(tmpdir)
        assert not df.empty
        # Check for new long-format columns
        for col in ["material_id", "components", "property", "value", "phase", "method"]:
            assert col in data.columns, f"Missing expected column: {col}"
        assert data["phase"].notnull().any(), "No phase information found in DataFrame"
        assert data["method"].notnull().any(), "No method information found in DataFrame"
        assert not any(c.startswith("prop_") for c in data.columns), "Found wide-format property columns in long-format DataFrame"
    finally:
        shutil.rmtree(tmpdir)
        
def test_parsed_content_correctness():
    filenames = [get_fn("je8006138.xml")]
    result = build_pandas_dataframe(filenames)
    data = result["data"]
    compounds = result["compounds"]
    repository_metadata = result["repository_metadata"]
    
    print("\n=== Columns in parsed DataFrame ===\n")
    print(list(data.columns))
    print("\n=== Data ===\n")
    print(data.head(3))
    print(f"Repository metadata: {repository_metadata}")
    
    assert not data.empty
    for col in ["material_id", "components", "property", "value", "phase", "method"]:
        assert col in data.columns, f"Missing expected column: {col}"
    assert data["phase"].notnull().any(), "No phase information found in DataFrame"
    assert data["method"].notnull().any(), "No method information found in DataFrame"
    assert not any(c.startswith("prop_") for c in data.columns), "Found wide-format property columns in long-format DataFrame"
    # Example: check for a known property/row if you have a reference value
    # known_row = data[(data["property"] == "Viscosity, Pa*s") & (data["value"] == 0.003881)]
    # assert not known_row.empty, "Expected property/value not found in DataFrame"


