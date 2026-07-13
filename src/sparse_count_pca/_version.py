"""Installed package-version discovery."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparse-count-pca")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
