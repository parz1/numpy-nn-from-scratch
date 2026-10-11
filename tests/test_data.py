import gzip
import hashlib
import io
import struct
import numpy as np
import pytest
from nn import data


def write_gzip(path, raw):
    path.write_bytes(gzip.compress(raw))
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_idx_images_and_labels(tmp_path):
    images = tmp_path / "images.gz"
    labels = tmp_path / "labels.gz"
    write_gzip(images, struct.pack(">IIII", 2051, 2, 2, 2) + bytes(range(8)))
    write_gzip(labels, struct.pack(">II", 2049, 2) + bytes([3, 9]))
    result = data.read_idx(images)
    assert result.dtype == np.uint8 and result.shape == (2, 2, 2)
    np.testing.assert_array_equal(result.ravel(), np.arange(8))
    np.testing.assert_array_equal(data.read_idx(labels), [3, 9])


@pytest.mark.parametrize("raw", [b"", struct.pack(">II", 999, 1),
    struct.pack(">II", 2051, 1), struct.pack(">II", 2049, 2) + b"\x01",
    struct.pack(">II", 2049, 1) + b"\x01\x02", struct.pack(">II", 2049, 0)])
def test_idx_bad_headers_or_lengths(tmp_path, raw):
    path = tmp_path / "bad.gz"
    write_gzip(path, raw)
    with pytest.raises(ValueError):
        data.read_idx(path)


def test_load_normalization_and_cache_offline(tmp_path, monkeypatch):
    checksums = {}
    for split in ("train", "t10k"):
        image_name, label_name = f"{split}-images-idx3-ubyte.gz", f"{split}-labels-idx1-ubyte.gz"
        checksums[image_name] = write_gzip(tmp_path / image_name,
            struct.pack(">IIII", 2051, 2, 28, 28) + bytes([0, 255]) * 784)
        checksums[label_name] = write_gzip(tmp_path / label_name,
            struct.pack(">II", 2049, 2) + bytes([1, 9]))
    monkeypatch.setattr(data, "FILES", checksums)
    def no_network(*args, **kwargs):
        pytest.fail("Cached/offline loading must not use network")
    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    train_x, train_y, test_x, test_y = data.load_mnist(tmp_path, download=False)
    for x, y in ((train_x, train_y), (test_x, test_y)):
        assert x.shape == (2, 784) and x.dtype == np.float32
        assert x.min() == 0 and x.max() == 1
        assert y.dtype == np.int64
        np.testing.assert_array_equal(y, [1, 9])
    # Valid caches are reused even when downloading is enabled.
    data.load_mnist(tmp_path, download=True)


def test_missing_offline_and_download_integrity(tmp_path, monkeypatch):
    name, payload = "sample.gz", b"fixture payload"
    monkeypatch.setattr(data, "FILES", {name: hashlib.md5(payload).hexdigest()})
    with pytest.raises(FileNotFoundError):
        data.download_mnist(tmp_path, download=False)
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    data.download_mnist(tmp_path)
    assert (tmp_path / name).read_bytes() == payload
    (tmp_path / name).write_bytes(b"corrupt cache")
    with pytest.raises(FileNotFoundError):
        data.download_mnist(tmp_path, download=False)
    data.download_mnist(tmp_path)
    assert (tmp_path / name).read_bytes() == payload
    (tmp_path / name).unlink()
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="checksum"):
        data.download_mnist(tmp_path)
    assert not (tmp_path / name).exists()
    assert not (tmp_path / (name + ".part")).exists()


def test_minibatches_alignment_shuffle_remainder():
    x, y = np.arange(14).reshape(7, 2), np.arange(7)
    batches = list(data.iter_minibatches(x, y, 3, shuffle=False))
    assert [len(labels) for _, labels in batches] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate([labels for _, labels in batches]), y)
    shuffled = list(data.iter_minibatches(x, y, 3, rng=np.random.default_rng(3)))
    repeated = list(data.iter_minibatches(x, y, 3, rng=np.random.default_rng(3)))
    all_labels = np.concatenate([labels for _, labels in shuffled])
    np.testing.assert_array_equal(np.sort(all_labels), y)
    np.testing.assert_array_equal(all_labels, np.concatenate([labels for _, labels in repeated]))
    for features, labels in shuffled:
        np.testing.assert_array_equal(features[:, 0], labels * 2)
    np.testing.assert_array_equal(x, np.arange(14).reshape(7, 2))


@pytest.mark.parametrize("x,y,batch", [(np.zeros((2, 1)), np.zeros(2), 0),
                                     (np.zeros((2, 1)), np.zeros(1), 2)])
def test_invalid_minibatches(x, y, batch):
    with pytest.raises(ValueError):
        list(data.iter_minibatches(x, y, batch))
