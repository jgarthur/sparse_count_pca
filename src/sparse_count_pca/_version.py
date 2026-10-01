"""Installed package-version discovery."""

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "sparse-count-pca"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
