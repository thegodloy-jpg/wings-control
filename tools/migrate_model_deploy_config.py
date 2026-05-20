#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""Migrate legacy *_default.json model_deploy_config to Phase D carriers.

The generated files live under:
  wings_control/config/recipes/model_deploy_compat/<device>.yaml

They intentionally preserve the legacy model_deploy_config shape as a
transition step.  Runtime reads can then move away from config/defaults JSON
before the data is fully decomposed into granular architecture/model recipes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "wings_control"))

from core.model_deploy_compat_loader import (  # noqa: E402
    build_compat_document,
    extract_model_deploy_config,
)


def write_structured_file(path: str | Path, data: dict[str, Any]) -> None:
    """Write deterministic JSON-subset YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_input(config_root: Path, device: str) -> Path:
    return config_root / "defaults" / f"{device}_default.json"


def _default_output(config_root: Path, device: str) -> Path:
    return config_root / "recipes" / "model_deploy_compat" / f"{device}.yaml"


def migrate_device(config_root: Path, device: str, *, dry_run: bool = False) -> dict[str, Any]:
    source = _default_input(config_root, device)
    target = _default_output(config_root, device)
    legacy = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    model_deploy_config = extract_model_deploy_config(legacy)
    try:
        source_ref = str(source.relative_to(config_root))
    except ValueError:
        source_ref = str(source)
    document = build_compat_document(device, model_deploy_config, source_ref=source_ref)
    stats = {
        "device": device,
        "source": str(source),
        "target": str(target),
        "model_types": len(model_deploy_config),
        "architecture_entries": sum(
            len(value) for value in model_deploy_config.values() if isinstance(value, dict)
        ),
    }
    if not dry_run:
        write_structured_file(target, document)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy model_deploy_config into Phase D carriers.")
    parser.add_argument("--config-root", default=str(_REPO_ROOT / "wings_control" / "config"))
    parser.add_argument("--device", choices=["nvidia", "ascend"], action="append", help="Device to migrate; repeatable. Defaults to both.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    devices = args.device or ["nvidia", "ascend"]
    results = [migrate_device(config_root, device, dry_run=args.dry_run) for device in devices]
    if args.output_json:
        print(json.dumps({"results": results, "dry_run": args.dry_run}, ensure_ascii=False, indent=2))
    else:
        action = "would write" if args.dry_run else "wrote"
        for item in results:
            print(
                f"{action} {item['target']} from {item['source']} "
                f"(model_types={item['model_types']}, entries={item['architecture_entries']})"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
