"""Lazy Project Aria VRS access for cluster-side RGB-D preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def is_numeric_stream_id(label: str) -> bool:
    """Return whether a stream selector has the ``recordable-type-index`` form."""
    return label.count("-") == 1 and all(part.isdigit() for part in label.split("-"))


class AriaVrsReader:
    """Read image/depth frames from one Aria VRS file by device time."""

    def __init__(self, path: str | Path) -> None:
        try:
            from projectaria_tools.core import data_provider
        except ImportError as exc:
            raise RuntimeError(
                "projectaria-tools is required for VRS assets; "
                "install it on the cluster"
            ) from exc
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._provider: Any = data_provider.create_vrs_data_provider(str(self.path))
        if self._provider is None:
            raise RuntimeError(f"cannot open VRS file: {self.path}")

    def stream_labels(self) -> tuple[str, ...]:
        """Return labels for all streams in the VRS file."""
        labels = [
            self._provider.get_label_from_stream_id(stream)
            for stream in self._provider.get_all_streams()
        ]
        return tuple(sorted(str(label) for label in labels if label))

    def image_at(self, label: str, timestamp_s: float) -> np.ndarray:
        """Return the image nearest before the causal timestamp."""
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId

        numeric_stream = is_numeric_stream_id(label)
        stream = (
            StreamId(label)
            if numeric_stream
            else self._provider.get_stream_id_from_label(label)
        )
        if stream is None:
            raise KeyError(f"VRS stream label not found: {label}")
        image_data = self._provider.get_image_data_by_time_ns(
            stream,
            int(round(timestamp_s * 1e9)),
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.BEFORE,
        )
        if image_data is None or image_data[0] is None:
            raise LookupError(f"no {label} frame at {timestamp_s:.6f} s")
        return np.asarray(image_data[0].to_numpy_array())


def find_vrs_file(asset_dir: str | Path) -> Path:
    """Find the single VRS file in an extracted ADT asset directory."""
    files = sorted(Path(asset_dir).rglob("*.vrs"))
    if not files:
        raise FileNotFoundError(f"no .vrs file under {asset_dir}")
    if len(files) > 1:
        raise ValueError(f"multiple VRS files under {asset_dir}; select one explicitly")
    return files[0]
