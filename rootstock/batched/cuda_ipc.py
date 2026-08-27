"""
CUDA IPC tensor sharing between the client and the batched worker.

Both processes sit on the same GPU, so tensors can be shared by memory
handle instead of copied over the socket: the exporter calls
``UntypedStorage._share_cuda_()`` and ships the resulting descriptor
(bytes base64-encoded into the wire header); the importer rebuilds a
live tensor over the same device memory with
``torch.multiprocessing.reductions.rebuild_cuda_tensor``. After that,
per-step traffic is a GPU-to-GPU ``copy_`` on one side and a tiny
control message — no serialization, no host round-trip.

Constraints: same machine, same visible CUDA device, and (unlike the
raw-bytes wire) a torch on both sides recent enough to share the
``rebuild_cuda_tensor`` signature. A descriptor can only be opened in a
*different* process from its exporter, which matches the client/worker
split exactly. The exporter must keep its tensor alive for as long as
the importer holds the rebuilt one.
"""

from __future__ import annotations

import base64
from typing import Any


def export_cuda_tensor(t) -> dict[str, Any]:
    """Descriptor for sharing ``t`` (a CUDA tensor) with another process."""

    if not t.is_cuda:
        raise ValueError("only CUDA tensors can be exported over CUDA IPC")
    storage = t.untyped_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "size": list(t.size()),
        "stride": list(t.stride()),
        "offset": t.storage_offset(),
        "device": device,
        "handle": base64.b64encode(handle).decode(),
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "ref_counter_handle": base64.b64encode(ref_counter_handle).decode(),
        "ref_counter_offset": ref_counter_offset,
        "event_handle": base64.b64encode(event_handle).decode(),
        "event_sync_required": event_sync_required,
    }


def import_cuda_tensor(desc: dict[str, Any]):
    """Rebuild a live tensor over the exporter's device memory."""
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    return rebuild_cuda_tensor(
        torch.Tensor,
        torch.Size(desc["size"]),
        tuple(desc["stride"]),
        desc["offset"],
        torch.UntypedStorage,
        getattr(torch, desc["dtype"]),
        desc["device"],
        base64.b64decode(desc["handle"]),
        desc["storage_size_bytes"],
        desc["storage_offset_bytes"],
        False,
        base64.b64decode(desc["ref_counter_handle"]),
        desc["ref_counter_offset"],
        base64.b64decode(desc["event_handle"]),
        desc["event_sync_required"],
    )
