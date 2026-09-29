from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from veriserve.common import atomic_json, load_config
from veriserve.data import prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/4060.yaml"))
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    asset_path = Path(".cache/assets.json")
    if not args.models_only:
        splits_path = Path(config.get("splits_path", ".cache/splits.json"))
        if config.get("require_frozen_splits") and not splits_path.exists():
            raise RuntimeError(f"Frozen split manifest is missing: {splits_path}")
        splits = prepare_splits(splits_path, int(config["seed"]))
        print({key: len(splits[key]) for key in ("debug", "development", "evaluation", "diagnostic")}, flush=True)
    if not args.data_only:
        api = HfApi()
        models = {}
        for role in ("generator", "verifier"):
            model_id = config[role]["id"]
            revision = api.model_info(model_id).sha
            print(f"Downloading {role}: {model_id}@{revision}", flush=True)
            location = snapshot_download(repo_id=model_id, revision=revision)
            models[role] = {"id": model_id, "revision": revision, "path": location}
            atomic_json(asset_path, models)
            print(f"Ready {role}: {location}", flush=True)


if __name__ == "__main__":
    main()
