"""NASA Earthdata access: authentication and streaming granule handles.

Imperative shell. Everything here touches the network or the filesystem,
which is why none of it lives in `processing` or `geospatial`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

PLACEHOLDER = "REPLACE_WITH"


class EarthdataError(RuntimeError):
    """Authentication or access failed in a way the operator can act on."""


def authenticate(netrc_path: Path) -> None:
    """Log in to Earthdata using the mounted netrc secret.

    earthaccess reads ~/.netrc, but the credential arrives as a read-only
    compose secret elsewhere, so it is copied into place first.
    """
    import earthaccess

    if not netrc_path.is_file():
        raise EarthdataError(
            f"No Earthdata credentials at {netrc_path}. "
            "Run 'make setup' and put your login in secrets/netrc."
        )
    if PLACEHOLDER in netrc_path.read_text(errors="ignore"):
        raise EarthdataError(
            "secrets/netrc still contains the placeholder. Register at "
            "https://urs.earthdata.nasa.gov/users/new and put the login in it."
        )

    home = Path(os.environ.get("HOME", "/tmp"))
    target = home / ".netrc"
    if target.resolve() != netrc_path.resolve():
        shutil.copyfile(netrc_path, target)
        target.chmod(0o600)

    auth = earthaccess.login(strategy="netrc", persist=False)
    if not getattr(auth, "authenticated", False):
        raise EarthdataError(
            "Earthdata rejected the credentials in secrets/netrc. "
            "Check the username and password at https://urs.earthdata.nasa.gov/"
        )


def search_granules(
    short_name: str,
    bounds: tuple[float, float, float, float],
    temporal: tuple[str, str],
    *,
    count: int = 200,
) -> list[Any]:
    """CMR granule search. Public API: no authentication required."""
    import earthaccess

    return earthaccess.search_data(
        short_name=short_name,
        bounding_box=bounds,
        temporal=temporal,
        count=count,
    )


def open_granule(granule: Any) -> Any:
    """A file-like object that fetches byte ranges on demand.

    Nothing is downloaded in full. h5py reads only the chunks the window
    touches, which is what makes a 5.7 GB granule usable on a laptop.
    """
    import earthaccess

    handles = earthaccess.open([granule])
    if not handles:
        raise EarthdataError(f"Could not open a stream for {granule}")
    return handles[0]
