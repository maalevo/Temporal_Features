from pathlib import Path
from typing import Any, Dict, List
import json
import time
import yaml
from datetime import datetime


from .experiment import Experiment, ExperimentResult


class ExperimentBatcher:
    """Runs multiple experiments defined in one YAML/JSON config file."""

    def __init__(self, config_path: str | Path, *, output_dir: str = "artifacts/batch", verbose: int = 1):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configs = self._load_config_file(self.config_path)

    def _load_config_file(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                data = yaml.safe_load(text)
            except Exception:
                data = json.loads(text)
        if not isinstance(data, dict) or "experiments" not in data:
            raise ValueError("Top-level key 'experiments' not found in config file.")
        return data["experiments"]

    def run_all(self) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        t0 = time.perf_counter()

        for i, cfg in enumerate(self.configs, start=1):
            cfg_name = cfg.get("name", f"experiment_{i}")
            print(f"\n[Batcher] Running experiment {i}/{len(self.configs)}: {cfg_name}")

            exp = Experiment(cfg, verbose=self.verbose)
            result = exp.run()
            results.append(result)

            self._save_result_summary(result, i)

        total_time = time.perf_counter() - t0
        print(f"\n[Batcher] ✅ Completed {len(results)} experiments in {total_time:.2f}s")
        return results

    def _save_result_summary(self, result: ExperimentResult, index: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            "index": index,
            "config_name": result.config.get("name", f"exp_{index}"),
            "dataset_path": result.config.get("dataset", {}).get("path", None),
            "artifact_path": result.artifact_path,
            "metrics": result.metrics,
            "duration_sec": result.duration_sec,
        }
        out_path = self.output_dir /  f"summary_{result.config.get('name', f'{index:02d}')}_{ts}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[Batcher] Saved summary → {out_path}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run a batch of experiments.")
    parser.add_argument("config_path", help="Path to YAML/JSON config containing multiple experiments.")
    parser.add_argument("--output_dir", default="artifacts/batch", help="Directory to save batch summaries.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0 = silent, 1 = normal).")
    args = parser.parse_args()

    batcher = ExperimentBatcher(args.config_path, output_dir=args.output_dir, verbose=args.verbose)
    batcher.run_all()
