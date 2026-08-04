"""Schema v6: shared installs (#208).

One manifest file per install, however many clusters mount it; verification
per cluster; the per-cluster split happens in the push payload. These tests
pin the v5 -> v6 migration semantics and the wire projection.
"""

from __future__ import annotations

from rootstock.clusters import Cluster, get_clusters_for_root
from rootstock.manifest import (
    SCHEMA_VERSION,
    CheckpointInfo,
    EnvironmentInfo,
    Maintainer,
    Manifest,
    VerificationRecord,
    manifest_push_payload,
    migrate_manifest_data,
)


def _v5(cluster="test", root="/tmp/x", environments=None) -> dict:
    return {
        "schema_version": 5,
        "cluster": cluster,
        "root": root,
        "maintainer": {"name": "a", "email": "a@b.c"},
        "rootstock_version": "1.3.0",
        "python_version": "3.11",
        "last_updated": "2026-01-01T00:00:00Z",
        "environments": environments or {},
    }


def _v5_env(checkpoints=None) -> dict:
    return {
        "built_at": "2026-01-01T00:00:00Z",
        "source_hash": "sha256:abc",
        "source": "CHECKPOINTS = {}",
        "python_requires": ">=3.11",
        "dependencies": {"mace-torch": "0.3.6"},
        "lock_hash": None,
        "checkpoints": checkpoints or {},
    }


# --- v5 -> v6 migration -----------------------------------------------------


def test_v5_flat_verification_moves_under_old_cluster():
    ckpt = {
        "fetched_at": "2026-01-02T00:00:00Z",
        "verified_at": "2026-01-03T00:00:00Z",
        "verified_device": "cuda",
        "last_error": None,
    }
    data = _v5(environments={"mace": _v5_env({"mace-mp-0-medium": ckpt})})

    migrated, _ = migrate_manifest_data(data)

    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["clusters"] == ["test"]
    assert "cluster" not in migrated
    out = migrated["environments"]["mace"]["checkpoints"]["mace-mp-0-medium"]
    assert out["fetched_at"] == "2026-01-02T00:00:00Z"
    assert "verified_at" not in out
    assert out["verifications"]["test"] == {
        "verified_at": "2026-01-03T00:00:00Z",
        "verified_device": "cuda",
        "last_error": None,
    }
    assert migrated["environments"]["mace"]["clusters"] is None


def test_v5_download_error_stays_shared_verify_error_moves():
    fetch_failed = {"fetched_at": None, "last_error": "download: no route to host"}
    verify_failed = {
        "fetched_at": "2026-01-02T00:00:00Z",
        "verified_at": None,
        "verified_device": None,
        "last_error": "smoke-test: CUDA OOM",
    }
    data = _v5(environments={"mace": _v5_env({"a": fetch_failed, "b": verify_failed})})

    migrated, _ = migrate_manifest_data(data)

    a = migrated["environments"]["mace"]["checkpoints"]["a"]
    assert a["last_error"] == "download: no route to host"
    assert a["verifications"] == {}

    b = migrated["environments"]["mace"]["checkpoints"]["b"]
    assert b["last_error"] is None
    assert b["verifications"]["test"]["last_error"] == "smoke-test: CUDA OOM"


def test_v5_shared_registry_root_seeds_sibling_clusters(monkeypatch):
    from pathlib import Path

    import rootstock.clusters as clusters_module

    registry = {
        "alpha": Cluster(root=Path("/shared/rootstock")),
        "beta": Cluster(root=Path("/shared/rootstock")),
        "gamma": Cluster(root=Path("/elsewhere")),
    }
    monkeypatch.setattr(clusters_module, "CLUSTER_REGISTRY", registry)

    data = _v5(cluster="beta", root="/shared/rootstock")
    migrated, notes = migrate_manifest_data(data)

    # The old identity stays first; the sibling is appended, not guessed at.
    assert migrated["clusters"] == ["beta", "alpha"]
    assert any("alpha" in n for n in notes)


def test_v5_unregistered_root_migrates_to_single_cluster():
    migrated, notes = migrate_manifest_data(_v5(cluster="della", root="/nowhere"))
    assert migrated["clusters"] == ["della"]
    # No sibling seeding — nothing extra to announce.
    assert notes == ["migrated manifest schema v5 -> v6"]


def test_migrated_v5_round_trips_through_dataclasses():
    ckpt = {
        "fetched_at": "2026-01-02T00:00:00Z",
        "verified_at": "2026-01-03T00:00:00Z",
        "verified_device": "cuda",
        "last_error": None,
    }
    data = _v5(environments={"mace": _v5_env({"mace-mp-0-medium": ckpt})})

    manifest = Manifest.from_dict(data)

    assert manifest.clusters == ["test"]
    record = manifest.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert record.verification("test").verified_at == "2026-01-03T00:00:00Z"
    assert record.verification("other").verified_at is None  # empty default

    again = Manifest.from_dict(manifest.to_dict())
    assert again.to_dict() == manifest.to_dict()


# --- push payload projection --------------------------------------------------


def _v6_manifest() -> Manifest:
    """Universal mace (medium verified on sophia, small failed on polaris)
    plus a polaris-only variant that records medium — the shadowing case."""
    return Manifest(
        schema_version=SCHEMA_VERSION,
        clusters=["sophia", "polaris"],
        root="/shared/rootstock",
        maintainer=Maintainer(name="a", email="a@b.c"),
        rootstock_version="1.4.0",
        python_version="3.11",
        last_updated="2026-08-01T00:00:00Z",
        environments={
            "mace": EnvironmentInfo(
                built_at="2026-01-01T00:00:00Z",
                source_hash="sha256:abc",
                source="",
                python_requires=">=3.11",
                dependencies={},
                checkpoints={
                    "mace-mp-0-medium": CheckpointInfo(
                        fetched_at="2026-01-02T00:00:00Z",
                        verifications={
                            "sophia": VerificationRecord(
                                verified_at="2026-01-03T00:00:00Z", verified_device="cuda"
                            ),
                        },
                    ),
                    "mace-mp-0-small": CheckpointInfo(
                        fetched_at="2026-01-02T00:00:00Z",
                        verifications={
                            "polaris": VerificationRecord(last_error="smoke-test: boom"),
                        },
                    ),
                },
            ),
            "mace-polaris": EnvironmentInfo(
                built_at="2026-01-01T00:00:00Z",
                source_hash="sha256:def",
                source="",
                python_requires=">=3.11",
                dependencies={},
                clusters=["polaris"],
                checkpoints={
                    "mace-mp-0-medium": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
                },
            ),
        },
    )


def test_push_payload_is_flat_per_cluster_v5():
    payload = manifest_push_payload(_v6_manifest(), "sophia")

    # The wire keeps the pre-v6 flat single-cluster shape: the backend files
    # by payload["cluster"] and the almanac reads flat verified_* fields.
    assert payload["schema_version"] == 5
    assert payload["cluster"] == "sophia"
    assert "clusters" not in payload

    ckpt = payload["environments"]["mace"]["checkpoints"]["mace-mp-0-medium"]
    assert ckpt["verified_at"] == "2026-01-03T00:00:00Z"
    assert ckpt["verified_device"] == "cuda"
    assert "verifications" not in ckpt


def test_push_payload_carries_only_that_clusters_results():
    payload = manifest_push_payload(_v6_manifest(), "polaris")
    ckpt = payload["environments"]["mace"]["checkpoints"]["mace-mp-0-small"]
    assert ckpt["verified_at"] is None
    assert ckpt["last_error"] == "smoke-test: boom"


def test_push_payload_omits_envs_not_serving_the_cluster():
    manifest = _v6_manifest()
    sophia = manifest_push_payload(manifest, "sophia")
    polaris = manifest_push_payload(manifest, "polaris")

    assert set(sophia["environments"]) == {"mace"}
    assert set(polaris["environments"]) == {"mace", "mace-polaris"}
    # The restriction is install-internal routing, not wire vocabulary.
    assert "clusters" not in polaris["environments"]["mace-polaris"]


def test_push_payload_verify_error_beats_shared_download_error():
    manifest = _v6_manifest()
    ckpt = manifest.environments["mace"].checkpoints["mace-mp-0-small"]
    ckpt.last_error = "download: flaky mirror"

    sophia = manifest_push_payload(manifest, "sophia")["environments"]["mace"]["checkpoints"]
    polaris = manifest_push_payload(manifest, "polaris")["environments"]["mace"]["checkpoints"]

    # sophia has no verify error -> its payload falls back to the shared fetch
    # error; polaris has its own verify error, which wins.
    assert sophia["mace-mp-0-small"]["last_error"] == "download: flaky mirror"
    assert polaris["mace-mp-0-small"]["last_error"] == "smoke-test: boom"


# --- per-id shadowing in the projection (#208, checkpoint-first) ---------------


def test_push_payload_variant_shadows_universal_per_id():
    manifest = _v6_manifest()
    polaris = manifest_push_payload(manifest, "polaris")
    sophia = manifest_push_payload(manifest, "sophia")

    # polaris: the overridden id is listed under the variant only; the id the
    # variant doesn't record stays under the universal env.
    assert "mace-mp-0-medium" not in polaris["environments"]["mace"]["checkpoints"]
    assert "mace-mp-0-medium" in polaris["environments"]["mace-polaris"]["checkpoints"]
    assert "mace-mp-0-small" in polaris["environments"]["mace"]["checkpoints"]
    # sophia's payload is untouched — the variant doesn't serve it.
    assert "mace-mp-0-medium" in sophia["environments"]["mace"]["checkpoints"]
    assert "mace-mp-0-small" in sophia["environments"]["mace"]["checkpoints"]


def test_push_payload_shadowing_reads_records_not_declarations():
    # Before the first checkpoint-first run writes the variant's record, the
    # universal row must stay — dropping it with nothing to show under the
    # variant would hide the id from the almanac entirely.
    manifest = _v6_manifest()
    manifest.environments["mace-polaris"].checkpoints.clear()
    polaris = manifest_push_payload(manifest, "polaris")
    assert "mace-mp-0-medium" in polaris["environments"]["mace"]["checkpoints"]


def test_push_payload_equal_specificity_never_shadows():
    # Two variants both serving polaris and recording the same id: an
    # authoring error resolution rejects loudly — keep both rows rather than
    # silently dropping one.
    manifest = _v6_manifest()
    manifest.environments["mace-polaris-b"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:ghi",
        source="",
        python_requires=">=3.11",
        dependencies={},
        clusters=["polaris"],
        checkpoints={"mace-mp-0-medium": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z")},
    )
    polaris = manifest_push_payload(manifest, "polaris")
    assert "mace-mp-0-medium" in polaris["environments"]["mace-polaris"]["checkpoints"]
    assert "mace-mp-0-medium" in polaris["environments"]["mace-polaris-b"]["checkpoints"]


def test_client_pushes_one_payload_per_cluster(monkeypatch):
    from rootstock.client import RootstockClient
    from rootstock.config import UserConfig

    posted: list[dict] = []

    def fake_post(self, url, payload, success_message):
        posted.append(payload)
        return True, success_message

    monkeypatch.setattr(RootstockClient, "_post", fake_post)
    config = UserConfig(api_key="k", api_secret="s", api_url="http://x")

    ok, message = RootstockClient(config).push_manifest(_v6_manifest())

    assert ok
    assert [p["cluster"] for p in posted] == ["sophia", "polaris"]
    assert "sophia: ok" in message and "polaris: ok" in message


def test_get_clusters_for_root_matches_shared_roots(monkeypatch):
    from pathlib import Path

    import rootstock.clusters as clusters_module

    registry = {
        "alpha": Cluster(root=Path("/shared/rootstock")),
        "beta": Cluster(root=Path("/shared/rootstock")),
    }
    monkeypatch.setattr(clusters_module, "CLUSTER_REGISTRY", registry)

    assert get_clusters_for_root("/shared/rootstock") == ["alpha", "beta"]
    assert get_clusters_for_root("/nowhere") == []
