from importlib.metadata import version as _version

from anops.annotation import identify_structure_and_annotate_gdx
from anops.plotting import plot_gdx

try:
    __version__ = _version("anops")
except Exception:
    __version__ = "9999"
