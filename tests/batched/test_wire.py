"""Wire protocol round-trip tests (numpy only; torch helpers where available)."""

import socket
import threading

import numpy as np
import pytest

from rootstock.batched.wire import MAGIC, WireError, recv_msg, send_msg


def _socketpair():
    return socket.socketpair()


def roundtrip(header, arrays):
    a, b = _socketpair()
    result = {}

    def receiver():
        result["msg"] = recv_msg(b)

    t = threading.Thread(target=receiver)
    t.start()
    sent = send_msg(a, header, arrays)
    t.join()
    a.close()
    b.close()
    got_header, got_arrays, received = result["msg"]
    assert received == sent
    return got_header, got_arrays


def test_roundtrip_tensors_and_metadata():
    arrays = {
        "positions": np.random.RandomState(0).randn(7, 3),
        "atomic_numbers": np.arange(7, dtype=np.int64),
        "flags": np.array([True, False, True]),
        "counts": np.array([3, 4], dtype=np.int32),
    }
    header, got = roundtrip({"type": "compute", "active_outputs": ["energy"]}, arrays)
    assert header["type"] == "compute"
    assert header["active_outputs"] == ["energy"]
    for name, arr in arrays.items():
        assert got[name].dtype == arr.dtype
        assert got[name].shape == arr.shape
        np.testing.assert_array_equal(got[name], arr)


def test_empty_tensor_set():
    header, got = roundtrip({"type": "shutdown"}, {})
    assert header["type"] == "shutdown"
    assert got == {}


def test_zero_length_tensor():
    header, got = roundtrip({"type": "compute"}, {"empty": np.zeros((0, 3))})
    assert got["empty"].shape == (0, 3)


def test_unsupported_dtype_rejected():
    a, b = _socketpair()
    with pytest.raises(WireError):
        send_msg(a, {"type": "compute"}, {"bad": np.zeros(2, dtype=np.complex128)})
    a.close()
    b.close()


def test_bad_magic_rejected():
    a, b = _socketpair()
    a.sendall(b"XXXX" + b"\x00" * 16)
    with pytest.raises(WireError):
        recv_msg(b)
    a.close()
    b.close()
    assert MAGIC != b"XXXX"


def test_torch_helpers_roundtrip():
    torch = pytest.importorskip("torch")
    from rootstock.batched.wire import arrays_to_tensors, tensors_to_arrays

    tensors = {
        "x": torch.randn(5, 3, dtype=torch.float32),
        "z": torch.arange(5),
        "b": torch.randn(4, dtype=torch.bfloat16),
    }
    arrays, bf16 = tensors_to_arrays(tensors)
    assert bf16 == ["b"]
    header, got = roundtrip({"type": "result", "_bf16": bf16}, arrays)
    back = arrays_to_tensors(header, got, device="cpu")
    assert back["b"].dtype == torch.bfloat16
    assert torch.equal(back["x"], tensors["x"])
    assert torch.equal(back["z"], tensors["z"])
    assert torch.equal(back["b"], tensors["b"])
