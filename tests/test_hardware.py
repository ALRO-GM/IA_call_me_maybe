"""Tests of the CPU thread selection."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src import hardware


def fake_sysfs(
    tmp_path: Path, cores: str, layout: Dict[int, Tuple[int, int]]
) -> None:
    """Write a fake ``/sys`` tree: CPU number -> (package, core id)."""
    (tmp_path / "cpus").write_text(cores)
    for cpu, (package, core) in layout.items():
        topology = tmp_path / f"cpu{cpu}"
        topology.mkdir()
        (topology / "physical_package_id").write_text(f"{package}\n")
        (topology / "core_id").write_text(f"{core}\n")


@pytest.fixture
def sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hardware, "PERFORMANCE_CPUS", str(tmp_path / "cpus"))
    monkeypatch.setattr(
        hardware, "TOPOLOGY", str(tmp_path / "cpu{}" / "{}")
    )
    return tmp_path


@pytest.mark.parametrize("text, expected", [
    ("0-3", [0, 1, 2, 3]),
    ("0,2,4-5\n", [0, 2, 4, 5]),
    ("7", [7]),
])
def test_parse_cpu_list(text: str, expected: List[int]) -> None:
    assert hardware.parse_cpu_list(text) == expected


def test_hyper_threads_of_one_core_count_once(sysfs: Path) -> None:
    # 4 logical CPUs on 2 physical performance cores.
    fake_sysfs(sysfs, "0-3", {0: (0, 0), 1: (0, 0), 2: (0, 4), 3: (0, 4)})
    assert hardware.performance_core_count() == 2


def test_no_hybrid_processor(sysfs: Path) -> None:
    assert hardware.performance_core_count() is None
    assert hardware.configure_threads() is None
    assert "OMP_NUM_THREADS" not in os.environ


def test_auto_uses_the_performance_cores(sysfs: Path) -> None:
    fake_sysfs(sysfs, "0-1", {0: (0, 0), 1: (0, 1)})
    assert hardware.configure_threads() == 2
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_explicit_count_wins_over_detection(sysfs: Path) -> None:
    fake_sysfs(sysfs, "0-1", {0: (0, 0), 1: (0, 1)})
    assert hardware.configure_threads(5) == 5
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_user_environment_is_respected(
    sysfs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert hardware.configure_threads(5) is None
    assert os.environ["OMP_NUM_THREADS"] == "3"
