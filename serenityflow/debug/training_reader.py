"""Read training metrics from SerenityBoard's SQLite database."""
from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = [
    "find_latest_run",
    "read_training_metrics",
]

_DEFAULT_TAGS = [
    "loss/train", "loss/avg", "grad_norm", "lr/default",
    "perf/steps_per_sec", "vram/allocated_gb", "vram/reserved_gb", "ema_decay",
]


def find_latest_run(log_dir: str) -> tuple[str, str] | None:
    """Find the most recently modified board.db in log_dir subdirectories.

    Returns (run_name, db_path) or None if no board.db found.
    Scans log_dir/*/board.db pattern.
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return None

    candidates = []
    for db_file in log_path.glob("*/board.db"):
        candidates.append((db_file.stat().st_mtime, db_file.parent.name, str(db_file)))

    if not candidates:
        return None

    candidates.sort(reverse=True)  # most recent first
    return candidates[0][1], candidates[0][2]


def _open_readonly(db_path: str) -> sqlite3.Connection:
    """Open SQLite DB in read-only mode with WAL support."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _get_session_status(conn: sqlite3.Connection) -> dict:
    """Get latest session info."""
    row = conn.execute(
        "SELECT session_id, start_time, resume_step, status "
        "FROM sessions ORDER BY start_time DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"session_id": None, "status": "unknown"}
    return {
        "session_id": row[0],
        "start_time": row[1],
        "resume_step": row[2],
        "status": row[3],
    }


def _read_latest_scalars(conn: sqlite3.Connection, tags: list[str]) -> dict:
    """Read the latest value for each tag."""
    result = {}
    for tag in tags:
        row = conn.execute(
            "SELECT step, wall_time, value FROM scalars WHERE tag = ? ORDER BY step DESC LIMIT 1",
            (tag,),
        ).fetchone()
        if row is not None:
            result[tag] = {"step": row[0], "wall_time": row[1], "value": round(row[2], 6)}
    return result


def _read_recent_series(conn: sqlite3.Connection, tag: str, n: int) -> list[list]:
    """Read last N data points for a tag. Returns [[step, value], ...]."""
    rows = conn.execute(
        "SELECT step, value FROM scalars WHERE tag = ? ORDER BY step DESC LIMIT ?",
        (tag, n),
    ).fetchall()
    # Reverse to chronological order
    return [[r[0], round(r[1], 6)] for r in reversed(rows)]


def _compute_summary(series_data: dict[str, list[list]]) -> dict:
    """Compute summary statistics from recent series."""
    summary = {}

    # Loss trend
    loss_series = series_data.get("loss/train") or series_data.get("loss/avg")
    if loss_series and len(loss_series) >= 2:
        values = [p[1] for p in loss_series]
        n = len(values)
        # Simple linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0

        summary["loss_trend"] = "decreasing" if slope < -1e-6 else ("increasing" if slope > 1e-6 else "stable")
        summary["loss_last_n_mean"] = round(y_mean, 6)
        summary["loss_last_n_std"] = round((sum((v - y_mean) ** 2 for v in values) / n) ** 0.5, 6)

    # Training speed
    speed_series = series_data.get("perf/steps_per_sec")
    if speed_series:
        speeds = [p[1] for p in speed_series]
        summary["training_speed_steps_per_sec"] = round(sum(speeds) / len(speeds), 4)

    # Grad norm stats
    grad_series = series_data.get("grad_norm")
    if grad_series:
        norms = [p[1] for p in grad_series]
        summary["grad_norm_mean"] = round(sum(norms) / len(norms), 6)
        summary["grad_norm_max"] = round(max(norms), 6)

    return summary


def _get_available_tags(conn: sqlite3.Connection) -> list[str]:
    """List all unique scalar tags."""
    rows = conn.execute("SELECT DISTINCT tag FROM scalars ORDER BY tag").fetchall()
    return [r[0] for r in rows]


def _get_available_runs(log_dir: str) -> list[str]:
    """List all run names in the log directory."""
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return []
    return sorted(
        d.name for d in log_path.iterdir()
        if d.is_dir() and (d / "board.db").exists()
    )


def read_training_metrics(
    log_dir: str,
    run_name: str | None = None,
    last_n_steps: int = 50,
    tags: list[str] | None = None,
) -> dict:
    """Read training metrics from a SerenityBoard database.

    Args:
        log_dir: Directory containing run subdirectories with board.db files.
        run_name: Specific run name. Auto-detects latest if None.
        last_n_steps: Number of recent data points for series.
        tags: Scalar tags to query. Uses defaults if None.

    Returns:
        Dict with run_name, session_status, latest, recent_series, summary, etc.

    Raises:
        FileNotFoundError: If no board.db found.
    """
    if tags is None:
        tags = list(_DEFAULT_TAGS)

    if run_name:
        db_path = str(Path(log_dir) / run_name / "board.db")
        if not Path(db_path).exists():
            raise FileNotFoundError(f"No board.db found at {db_path}")
    else:
        found = find_latest_run(log_dir)
        if found is None:
            raise FileNotFoundError(f"No board.db found in {log_dir}")
        run_name, db_path = found

    conn = _open_readonly(db_path)
    try:
        session = _get_session_status(conn)
        latest = _read_latest_scalars(conn, tags)

        # Get current step from highest step across all queried tags
        current_step = max((v["step"] for v in latest.values()), default=0)

        # Read recent series for trending tags
        series_tags = ["loss/train", "loss/avg", "grad_norm", "perf/steps_per_sec"]
        recent_series = {}
        for tag in series_tags:
            series = _read_recent_series(conn, tag, last_n_steps)
            if series:
                recent_series[tag] = series

        summary = _compute_summary(recent_series)
        available_tags = _get_available_tags(conn)

        return {
            "run_name": run_name,
            "run_dir": str(Path(log_dir) / run_name),
            "session_status": session.get("status", "unknown"),
            "current_step": current_step,
            "latest": latest,
            "recent_series": recent_series,
            "summary": summary,
            "available_tags": available_tags,
            "available_runs": _get_available_runs(log_dir),
        }
    finally:
        conn.close()
