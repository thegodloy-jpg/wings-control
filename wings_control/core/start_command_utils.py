"""Shared helpers for writing engine start scripts to the pod volume."""

from __future__ import annotations

import os
import stat

from config.settings import settings
from utils.file_utils import WriteOptions, safe_write_file


def write_start_command(script_text: str) -> str:
    """Write ``start_command.sh`` with cross-container executable permissions."""
    shared_dir = settings.SHARED_VOLUME_PATH
    os.makedirs(shared_dir, exist_ok=True)
    path = os.path.join(shared_dir, settings.START_COMMAND_FILENAME)
    ok = safe_write_file(
        path,
        script_text,
        is_json=False,
        options=WriteOptions(
            # Keep the script executable so deployments can use either
            # `bash /shared-volume/start_command.sh` or direct execution.
            modes=(
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                | stat.S_IRGRP | stat.S_IXGRP
                | stat.S_IROTH | stat.S_IXOTH
            ),
            atomic=True,
        ),
    )
    if not ok:
        raise RuntimeError(f"failed to write start command: {path}")
    return path
