"""MNIST download, integrity checks, IDX parsing and minibatches (complete)."""

import gzip
import hashlib
from pathlib import Path
import struct
import urllib.request

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # Dataset integrity fingerprint, not cryptography.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mnist(data_dir: str | Path, *, download: bool = True) -> Path:
    """Reuse checked files; download missing/corrupt files atomically.

    download=False never uses the network and raises for missing/corrupt files.
    Network/checksum errors leave no partially downloaded final file.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = root / name
        if destination.is_file() and _md5(destination) == expected:
            continue
        if not download:
            raise FileNotFoundError(f"Missing or corrupt MNIST file: {destination}")
        temporary = root / (name + ".part")
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
                with temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            if _md5(temporary) != expected:
                raise ValueError(f"MNIST checksum mismatch: {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return root


def read_idx(path: str | Path) -> np.ndarray:
    """Read uint8 gzip IDX images (2051) or labels (2049); reject bad payloads."""
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if len(raw) < 8:
        raise ValueError("Truncated IDX header")
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2051:
        if len(raw) < 16:
            raise ValueError("Truncated IDX image header")
        rows, columns = struct.unpack(">II", raw[8:16])
        shape, offset = (count, rows, columns), 16
    elif magic == 2049:
        shape, offset = (count,), 8
    else:
        raise ValueError(f"Unsupported IDX magic number: {magic}")
    if any(size <= 0 for size in shape):
        raise ValueError("IDX dimensions must be positive")
    if len(raw) - offset != int(np.prod(shape)):
        raise ValueError("IDX payload length does not match header")
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape).copy()


def load_mnist(data_dir: str | Path = "data/mnist", *, download: bool = True
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train_x, train_y, test_x, test_y; x: float32 (N,784) in [0,1]."""
    root = download_mnist(data_dir, download=download)
    arrays = []
    for split in ("train", "t10k"):
        images = read_idx(root / f"{split}-images-idx3-ubyte.gz")
        labels = read_idx(root / f"{split}-labels-idx1-ubyte.gz")
        if (images.ndim != 3 or images.shape[1:] != (28, 28)
                or labels.ndim != 1 or images.shape[0] != labels.shape[0]):
            raise ValueError("MNIST must contain matched 28x28 images and labels")
        if np.any(labels > 9):
            raise ValueError("MNIST labels must be in [0,9]")
        arrays.extend((images.reshape(-1, 784).astype(np.float32) / 255.0,
                       labels.astype(np.int64)))
    return tuple(arrays)


def iter_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, *,
                     rng: np.random.Generator | None = None, shuffle: bool = True):
    """Yield aligned batches, including a final partial batch; do not mutate input."""
    if batch_size <= 0 or len(x) != len(y):
        raise ValueError("Require positive batch_size and matched sample counts")
    indices = np.arange(len(x))
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selection = indices[start:start + batch_size]
        yield x[selection], y[selection]
