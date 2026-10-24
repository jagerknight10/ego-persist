"""ADT manifest: sequence selection, sizes, and download plan.

The manifest is the ``Aria Digital Twin Dataset URLs.json`` file. It is
treated as input, never committed. The download plan it produces lists only
the assets a stage needs, with sizes, so cluster jobs can verify disk space
before downloading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetSpec:
    """One downloadable asset of one sequence."""

    sequence_id: str
    asset: str
    filename: str
    url: str
    sha1sum: str
    size_bytes: int


@dataclass(frozen=True)
class SequenceSpec:
    """One sequence and all of its asset specs."""

    sequence_id: str
    assets: dict[str, AssetSpec]

    @property
    def total_bytes(self) -> int:
        return sum(asset.size_bytes for asset in self.assets.values())


class AdtManifest:
    """Parsed ADT URL manifest."""

    def __init__(self, path: str | Path) -> None:
        raw = json.loads(Path(path).read_text())
        sequences = raw["sequences"]
        self.sequences: dict[str, SequenceSpec] = {}
        for sequence_id, assets in sequences.items():
            specs = {
                asset: AssetSpec(
                    sequence_id=sequence_id,
                    asset=asset,
                    filename=str(info["filename"]),
                    url=str(info["download_url"]),
                    sha1sum=str(info["sha1sum"]),
                    size_bytes=int(info["file_size_bytes"]),
                )
                for asset, info in assets.items()
            }
            self.sequences[sequence_id] = SequenceSpec(sequence_id, specs)

    def select(
        self,
        sequence_ids: list[str],
        assets: list[str] | None = None,
    ) -> list[AssetSpec]:
        """Flatten the download plan for the given sequences and assets."""
        assets = set(assets) if assets else None
        plan: list[AssetSpec] = []
        for sequence_id in sequence_ids:
            spec = self.sequences[sequence_id]
            for asset, item in sorted(spec.assets.items()):
                if assets is None or asset in assets:
                    plan.append(item)
        return plan

    def total_bytes(self, sequence_ids: list[str], assets: list[str] | None) -> int:
        return sum(item.size_bytes for item in self.select(sequence_ids, assets))
