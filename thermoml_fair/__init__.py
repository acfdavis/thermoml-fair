import importlib.metadata

try:
    __version__ = importlib.metadata.version("thermoml_fair")
except importlib.metadata.PackageNotFoundError:
    # This can happen if the package is not installed (e.g., when running from source
    # without an editable install, or if the package metadata is somehow corrupted).
    # Fallback to a sensible default or indicate that the version is unknown.
    __version__ = "0.0.0-unknown" # Or None, or raise an error, depending on desired behavior
