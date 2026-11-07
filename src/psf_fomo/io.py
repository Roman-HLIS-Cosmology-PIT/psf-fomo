"""Shared YAML helpers and provenance records."""
import datetime
import platform
from pathlib import Path

import numpy as np
import yaml


def to_plain(value):
    """``value`` with NumPy scalars and arrays turned into plain Python types, recursively.

    NumPy scalars → ``int`` / ``float`` / ``bool`` / ``str``; arrays and tuples → lists;
    dicts and lists are converted item by item. YAML's safe dumper accepts only plain types,
    and NumPy values reach configs easily (e.g. a catalog's ``meta``).
    """
    if isinstance(value, dict):
        return {to_plain(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_yaml(path, data):
    """Write a dict to a YAML file.

    Parameters
    ----------
    path : str or Path
        Output file name.
    data : dict
        Data of plain Python types (dict, list, str, int, float, bool, None); NumPy scalars,
        arrays and tuples are converted first (`to_plain`).
    """
    with open(path, "w") as f:
        yaml.safe_dump(to_plain(data), f, sort_keys=False)


def read_yaml(path):
    """Read a YAML file written by `write_yaml`.

    Parameters
    ----------
    path : str or Path
        Input file name.

    Returns
    -------
    dict
    """
    with open(path) as f:
        return yaml.safe_load(f)


def piff_commit():
    """Return the git commit of the installed Piff, or None if it cannot be determined.

    Piff's version string does not distinguish ``main`` from the release, so the commit is
    read from the ``.git`` directory next to the ``piff`` package. This only works when Piff
    is used from a git checkout (e.g. an editable install of the clone); otherwise None.
    No git command is run.
    """
    import piff

    git_dir = Path(piff.__file__).resolve().parent.parent / ".git"
    if not git_dir.is_dir():
        return None
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head  # detached HEAD: the commit itself
    ref = head.split()[1]
    if (git_dir / ref).is_file():
        return (git_dir / ref).read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return None


def provenance():
    """Versions of the software used to make an output, plus a creation time.

    Returns
    -------
    dict
        Keys: ``psf_fomo``, ``galsim``, ``piff``, ``piff_commit``, ``numpy``, ``python``,
        ``created`` (local time, ISO format).
    """
    import galsim
    import numpy
    import piff

    import psf_fomo

    return {
        "psf_fomo": psf_fomo.__version__,
        "galsim": galsim.__version__,
        "piff": piff.__version__,
        "piff_commit": piff_commit(),
        "numpy": numpy.__version__,
        "python": platform.python_version(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
