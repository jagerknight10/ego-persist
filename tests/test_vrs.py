"""Tests for VRS asset discovery without the optional Aria package."""

from __future__ import annotations

import pytest

from wise_egomap.data.vrs import find_vrs_file


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
