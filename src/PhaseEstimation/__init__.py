from importlib.metadata import metadata

PACKAGE = "PhaseEstimation"

__version__ = metadata(PACKAGE)["version"]
