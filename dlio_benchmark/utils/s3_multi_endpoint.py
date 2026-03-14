from __future__ import annotations
from typing import List, Optional
import os

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except Exception:
    _COMM = None
    _RANK = 0
    _SIZE = 1


def parse_endpoints(csv: Optional[str]) -> List[str]:
    """Split comma-separated endpoints into a trimmed list."""
    if not csv:
        return []
    return [e.strip() for e in str(csv).split(",") if e and str(e).strip()]


def select_endpoint_by_rank(endpoints: List[str], rank: Optional[int] = None) -> Optional[str]:
    """Return endpoints[rank % len(endpoints)] or None if empty."""
    if not endpoints:
        return None
    if rank is None:
        rank = _RANK
    return endpoints[rank % len(endpoints)]


def resolve_s3_endpoint(config_csv_value: Optional[str]) -> Optional[str]:
    """
    Resolve the S3 endpoint with the following precedence:

      1) DLIO_S3_ENDPOINTS (CSV), indexed by rank
      2) DLIO_S3_ENDPOINT (single)
      3) config_csv_value (from YAML/Hydra; may be CSV or single)
      4) None (fallback to library/default behavior)

    Notes:
    - The function preserves the original rank-based selection behavior.
    - `config_csv_value` is typically the value of
      cfg.workload.storage.storage_options.endpoint_url, if present.
    """
    env_csv = os.environ.get("DLIO_S3_ENDPOINTS")
    env_single = os.environ.get("DLIO_S3_ENDPOINT")

    # 1) CSV endpoints via env: DLIO_S3_ENDPOINTS
    endpoints_from_env_csv = parse_endpoints(env_csv)
    if endpoints_from_env_csv:
        return select_endpoint_by_rank(endpoints_from_env_csv, _RANK)

    # 2) Single endpoint via env: DLIO_S3_ENDPOINT
    if env_single and env_single.strip():
        return env_single.strip()

    # 3) YAML/Hydra-provided value (could be CSV or single)
    endpoints_from_cfg = parse_endpoints(config_csv_value)
    if endpoints_from_cfg:
        return select_endpoint_by_rank(endpoints_from_cfg, _RANK)

    # 4) Fallback (let downstream S3 client use its default behavior)
    return None
