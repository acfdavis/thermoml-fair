import pytest
import pandas as pd
from pathlib import Path
from thermoml_fair.core.utils import build_pandas_dataframe

# A minimal, valid ThermoML XML string for testing purposes.
# It contains one property (Boiling point) measured at a specific temperature and pressure.
SAMPLE_XML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<ThermoML xmlns="http://www.iupac.org/namespaces/ThermoML" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Citation>
    <sAuthor>Test Author</sAuthor>
    <sTitle>Test Data</sTitle>
    <sJournal>Test Journal</sJournal>
    <yrPubYr>2025</yrPubYr>
    <sDOI>10.1000/test</sDOI>
  </Citation>
  <Compound>
    <RegNum>
      <nOrgNum>1</nOrgNum>
    </RegNum>
    <sStandardInChI>InChI=1S/H2O/h1H2</sStandardInChI>
    <sStandardInChIKey>XLYOFNOQVPJJNP-UHFFFAOYSA-N</sStandardInChIKey>
    <sCommonName>water</sCommonName>
    <sFormulaMolec>H2O</sFormulaMolec>
  </Compound>
  <PureOrMixtureData>
    <nPureOrMixtureData>1</nPureOrMixtureData>
    <Component>
      <RegNum>
        <nOrgNum>1</nOrgNum>
      </RegNum>
      <nSampleAbs>1</nSampleAbs>
    </Component>
    <Property>
      <nPropNumber>1</nPropNumber>
      <Property-MethodID>
        <PropertyGroup>
          <ThermodynProp>
            <ePropName>Boiling temperature, K</ePropName>
            <eMethodName>Ebulliometric method</eMethodName>
          </ThermodynProp>
        </PropertyGroup>
      </Property-MethodID>
      <PropPhaseID>
        <ePropPhase>Liquid</ePropPhase>
      </PropPhaseID>
      <ePresentation>Direct value, X</ePresentation>
    </Property>
    <Variable>
      <nVarNumber>1</nVarNumber>
      <VariableID>
        <VariableType>
            <eVarType>Pressure, kPa</eVarType>
        </VariableType>
      </VariableID>
      <VarPhaseID>
        <eVarPhase>Liquid</eVarPhase>
      </VarPhaseID>
    </Variable>
    <NumValues>
      <VariableValue>
        <nVarNumber>1</nVarNumber>
        <nVarValue>101.325</nVarValue>
      </VariableValue>
      <PropertyValue>
        <nPropNumber>1</nPropNumber>
        <nPropValue>373.15</nPropValue>
        <nPropDigits>4</nPropDigits>
      </PropertyValue>
    </NumValues>
  </PureOrMixtureData>
</ThermoML>
"""

@pytest.fixture
def sample_xml_file(tmp_path: Path) -> str:
    """Create a temporary XML file with sample content for testing."""
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text(SAMPLE_XML_CONTENT)
    return str(xml_file)

def test_build_dataframe_includes_variables(sample_xml_file: str):
    """
    Tests that the build_pandas_dataframe function correctly creates a DataFrame
    that includes columns for experimental variables (e.g., Pressure).
    This test will fail until the core logic is updated.
    """
    # Call the function to build the DataFrame from the test XML
    # We pass xsd_path_or_obj=None to skip schema validation for this unit test.
    # The purpose of this test is to check the DataFrame creation logic,
    # and schema validity is already covered by tests using real data files.
    data_bundle = build_pandas_dataframe(xml_files=[sample_xml_file], xsd_path_or_obj=None)
    df = data_bundle.get('data')

    # --- Assertions ---
    assert df is not None, "The DataFrame should not be None."
    assert not df.empty, "The DataFrame should not be empty."

    # Define the expected column name for the variable
    # The function should create a column named 'Pressure, kPa' (no 'var_' prefix)
    expected_variable_column = 'Pressure, kPa'

    # 1. Check if the variable column exists
    assert expected_variable_column in df.columns, (
        f"The output DataFrame is missing the expected variable column: '{expected_variable_column}'. "
        f"Available columns: {list(df.columns)}"
    )

    # 2. Check if the value in the variable column is correct
    # The value should be a float, not a string.
    actual_value = df[expected_variable_column].iloc[0]
    expected_value = 101.325

    assert pd.api.types.is_numeric_dtype(df[expected_variable_column].dtype), (
        f"The variable column '{expected_variable_column}' should be a numeric type, but it is {df[expected_variable_column].dtype}."
    )
    
    assert actual_value == pytest.approx(expected_value), (
        f"The value in the '{expected_variable_column}' column is incorrect. "
        f"Expected: {expected_value}, Found: {actual_value}"
    )

    # 3. Check for correct property name and value
    assert df['property'].iloc[0] == 'Boiling temperature, K'
    assert df['value'].iloc[0] == pytest.approx(373.15)
