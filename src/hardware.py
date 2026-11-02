"""Choosing how many CPU threads the model runs on.

On a hybrid processor (Intel 12th generation and later) the default thread
count of the model library includes the slow efficiency cores. Every matrix
product is split evenly between the threads, so the fast cores end up
waiting for the slow ones: on an i7-12700KF one model call takes 0.62 s with
the default 12 threads and 0.32 s with the 8 performance cores only.

The thread count is passed through ``OMP_NUM_THREADS``, which the library
reads when it is first imported, so it must be set before loading the SDK.
"""

import os
from typing import List, Optional, Set, Tuple

THREADS_VARIABLE = "OMP_NUM_THREADS"
PERFORMANCE_CPUS = "/sys/devices/cpu_core/cpus"
TOPOLOGY = "/sys/devices/system/cpu/cpu{}/topology/{}"


def parse_cpu_list(text: str) -> List[int]:
    """Expand a Linux CPU list such as ``0-3,8,10-11``.

    Args:
        text: The list, as written in ``/sys``.

    Returns:
        Every CPU number of the list.

    Raises:
        ValueError: If the text is not a CPU list.
    """
    cpus: List[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _read(path: str) -> str:
    """Return the content of a small text file.

    Args:
        path: File to read.

    Returns:
        The text of the file.

    Raises:
        OSError: If the file cannot be read.
    """
    with open(path, "r", encoding="ascii") as handle:
        return handle.read()


def performance_core_count() -> Optional[int]:
    """Count the physical performance cores of a hybrid processor.

    Linux describes a hybrid Intel processor with two CPU types,
    ``cpu_core`` (performance) and ``cpu_atom`` (efficiency). Two logical
    CPUs of the same physical core (hyper-threading) count once.

    Returns:
        The number of performance cores, or ``None`` when the processor is
        not hybrid or the information is not available.
    """
    try:
        cpus = parse_cpu_list(_read(PERFORMANCE_CPUS))
        cores: Set[Tuple[str, str]] = {
            (
                _read(TOPOLOGY.format(cpu, "physical_package_id")).strip(),
                _read(TOPOLOGY.format(cpu, "core_id")).strip(),
            )
            for cpu in cpus
        }
    except (OSError, ValueError):
        return None
    return len(cores) or None


def configure_threads(requested: int = 0) -> Optional[int]:
    """Set the number of CPU threads the model will use.

    An ``OMP_NUM_THREADS`` already defined by the user always wins. Must be
    called before the SDK is imported to have any effect.

    Args:
        requested: Thread count to use; 0 picks the performance cores of a
            hybrid processor and otherwise keeps the library default.

    Returns:
        The thread count that was set, or ``None`` if nothing was changed.
    """
    if THREADS_VARIABLE in os.environ:
        return None
    count = requested or performance_core_count()
    if not count:
        return None
    os.environ[THREADS_VARIABLE] = str(count)
    return count
