"""Tests for VRS asset discovery without the optional Aria package."""

from __future__ import annotations

import pytest

from wise_egomap.data.vrs import find_vrs_file, is_numeric_stream_id


def test_numeric_stream_id_detection() -> None:
    assert is_numeric_stream_id("345-1")
    assert is_numeric_stream_id("400-1")
    assert not is_numeric_stream_id("camera-rgb")
    assert not is_numeric_stream_id("345")


def test_find_vrs_file_requires_one_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        find_vrs_file(tmp_path)
    first = tmp_path / "first.vrs"
    second = tmp_path / "second.vrs"
    first.write_bytes(b"")
    assert find_vrs_file(tmp_path) == first
    second.write_bytes(b"")
    with pytest.raises(ValueError, match="multiple"):
        find_vrs_file(tmp_path)
