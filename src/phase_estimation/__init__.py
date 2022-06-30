from importlib.metadata import metadata

PACKAGE = "phase_estimation"

__version__ = metadata(PACKAGE)["version"]
