from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparse-residual-pca")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
