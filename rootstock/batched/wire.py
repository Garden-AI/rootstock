"""
Wire protocol for batched model serving.

Length-prefixed frames over a stream socket. Each frame is:

    MAGIC (4 bytes) | header length (uint32 LE) | header JSON | tensor payload

The header carries a message ``type``, arbitrary JSON metadata, and a
``tensors`` manifest — ``[{name, dtype, shape, nbytes}, ...]`` — describing
the raw buffers concatenated in the payload, in order.

Tensors travel as raw C-contiguous bytes plus dtype/shape, never as
pickles: the two endpoints run different Python environments with
different torch versions by design, and raw buffers are immune to that
skew. bfloat16 (no numpy equivalent) travels as uint16 bytes under a
``"bfloat16"`` dtype tag; the torch helpers reverse the view.

Message types (header["type"]):

- ``hello``    worker -> client, once after model load: the wrapper's
               ModelConfig JSON, the wire keys the worker wants per step,
               and worker diagnostics.
- ``compute``  client -> worker: one batch of named tensors + the active
               output set.
- ``result``   worker -> client: output tensors + worker-side timings.
- ``error``    worker -> client: traceback string; the worker stays alive.
- ``shutdown`` client -> worker; the worker acks with ``bye`` and exits.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

MAGIC = b"RSB1"
PROTOCOL_VERSION = 1

# numpy-representable dtypes, by tag. bfloat16 is handled in the torch
# helpers (uint16 buffer + this tag).
_TAG_TO_NP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "bool": np.bool_,
}
_NP_TO_TAG = {np.dtype(v): k for k, v in _TAG_TO_NP.items()}
BFLOAT16_TAG = "bfloat16"


class WireError(RuntimeError):
    """Malformed frame or closed socket mid-frame."""


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4 << 20))
        if not chunk:
            raise WireError(f"socket closed with {remaining} of {n} bytes unread")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> int:
    """Send one frame; ``arrays`` values must be numpy arrays. Returns bytes sent."""
    manifest = []
    buffers = []
    for name, arr in arrays.items():
        if arr.dtype not in _NP_TO_TAG:
            raise WireError(f"unsupported dtype {arr.dtype} for tensor {name!r}")
        arr = np.ascontiguousarray(arr)
        buf = arr.tobytes()
        # A bfloat16 tensor arrives here pre-viewed as uint16 with its tag
        # supplied via header["_bf16"] (see tensors_to_arrays).
        dtype_tag = _NP_TO_TAG[arr.dtype]
        if name in header.get("_bf16", ()):
            dtype_tag = BFLOAT16_TAG
        manifest.append(
            {"name": name, "dtype": dtype_tag, "shape": list(arr.shape), "nbytes": len(buf)}
        )
        buffers.append(buf)
    header = {k: v for k, v in header.items() if k != "_bf16"}
    header["tensors"] = manifest
    header_bytes = json.dumps(header).encode()
    frame = b"".join([MAGIC, struct.pack("<I", len(header_bytes)), header_bytes, *buffers])
    sock.sendall(frame)
    return len(frame)


def recv_msg(sock: socket.socket) -> tuple[dict[str, Any], dict[str, np.ndarray], int]:
    """Receive one frame. Returns (header, arrays, total bytes received)."""
    magic = _recv_exact(sock, 4)
    if magic != MAGIC:
        raise WireError(f"bad magic {magic!r}")
    (header_len,) = struct.unpack("<I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, header_len))
    arrays: dict[str, np.ndarray] = {}
    total = 8 + header_len
    for entry in header.get("tensors", []):
        buf = _recv_exact(sock, entry["nbytes"])
        total += entry["nbytes"]
        tag = entry["dtype"]
        np_dtype = _TAG_TO_NP["uint16" if tag == BFLOAT16_TAG else tag]
        arrays[entry["name"]] = np.frombuffer(buf, dtype=np_dtype).reshape(entry["shape"])
    return header, arrays, total


# ---------------------------------------------------------------------------
# torch <-> numpy helpers (torch imported lazily: the client package must
# stay importable in environments without torch)
# ---------------------------------------------------------------------------


def tensors_to_arrays(tensors: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Convert torch tensors to CPU numpy arrays.

    Returns (arrays, bf16_names); pass bf16_names as ``header["_bf16"]`` to
    send_msg so those buffers keep their bfloat16 identity on the wire.
    """
    import torch

    arrays: dict[str, np.ndarray] = {}
    bf16: list[str] = []
    for name, t in tensors.items():
        t = t.detach()
        if t.dtype == torch.bfloat16:
            t = t.contiguous().view(torch.uint16)
            bf16.append(name)
        arrays[name] = t.cpu().contiguous().numpy()
    return arrays, bf16


def arrays_to_tensors(
    header: dict[str, Any], arrays: dict[str, np.ndarray], device: Any
) -> dict[str, Any]:
    """Rehydrate wire arrays as torch tensors on ``device``."""
    import torch

    bf16_names = {e["name"] for e in header.get("tensors", []) if e["dtype"] == BFLOAT16_TAG}
    out: dict[str, Any] = {}
    for name, arr in arrays.items():
        # frombuffer arrays are read-only views; copy so torch owns writable memory.
        t = torch.from_numpy(np.array(arr))
        if name in bf16_names:
            t = t.view(torch.bfloat16)
        out[name] = t.to(device)
    return out
