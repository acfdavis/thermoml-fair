import os
from pathlib import Path

# Default base path for ThermoML data, caches, and archives.
# Users can override this by setting the THERMOML_PATH environment variable.
THERMOML_PATH = os.environ.get("THERMOML_PATH", "~/.thermoml")

# Default path to the ThermoML schema file.
# This can be overridden by the thermoml_fair_SCHEMA_PATH environment variable at runtime.
THERMOML_SCHEMA_PATH = str(Path(__file__).parent.parent / "data" / "ThermoML.xsd")

# Default subdirectory name within THERMOML_PATH for extracted XML files
DEFAULT_EXTRACTED_XML_DIR_NAME = "extracted_xml"

# Default subdirectory name for cached parsed files
DEFAULT_CACHE_DIR_NAME = "thermoml_cache"
DEFAULT_COMPOUNDS_FILE = "compounds.csv"
DEFAULT_DATA_FILE = "thermoml_data.csv"

# You can add other global configuration variables here as your project grows.
# For example:
# DEFAULT_SCHEMA_FILE_NAME = "ThermoML.xsd"
# API_TIMEOUT_SECONDS = 30
