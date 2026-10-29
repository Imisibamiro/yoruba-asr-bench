import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def extract_embeddings(audio_path, output_path, dataset_type, model, config, device):
    dataset = dataset_type(
        [(audio_path, output_path)],
        context_length=config["TRAIN"]["CONTEXT_LENGTH"],
        downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
    )
    cqt, _, _ = dataset[0]
    if cqt is None:
        raise RuntimeError

    embeddings = []
    with torch.inference_mode():
        whole = cqt.unsqueeze(0).unsqueeze(1).to(device)
        embeddings.extend(model(whole).cpu().numpy())

        window = int(
            config["TRAIN"]["CONTEXT_LENGTH"]
            // config["MODEL"]["DOWNSAMPLE_FACTOR"]
        )
        chunks = cqt.unfold(dimension=-1, size=window, step=window)
        chunks = chunks.permute(1, 0, 2).unsqueeze(1).to(device)
        embeddings.extend(model(chunks).cpu().numpy())

    return np.asarray(embeddings, dtype=np.float32)


def main():
    started = time.perf_counter()
    clip_paths = sorted(Path("tri/clips").glob("*.wav"))
    ref_paths = sorted(Path("tri/refs").glob("*.m4a"))

    try:
        vinet_root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(vinet_root))

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            import yaml
            from model.dataset import InferenceDataset
            from model.utils import load_model

            config_path = vinet_root / "logs/checkpoints/Discogs-VINet/config.yaml"
            with config_path.open() as file:
                config = yaml.safe_load(file)
            config["MODEL"]["CHECKPOINT_PATH"] = str(
                vinet_root / config["MODEL"]["CHECKPOINT_PATH"]
            )

            device = torch.device("cpu")
            model = load_model(config, device=device, mode="infer")
            model.eval()

            ref_embeddings = []
            ref_seconds = 0.0
            for index, ref_path in enumerate(ref_paths):
                item_started = time.perf_counter()
                output_path = Path("tri/.tune-embeddings") / f"{index}.npy"
                vector = extract_embeddings(
                    ref_path,
                    output_path,
                    InferenceDataset,
                    model,
                    config,
                    device,
                )[0]
                ref_embeddings.append(vector)
                ref_seconds += time.perf_counter() - item_started

            ref_matrix = np.stack(ref_embeddings)
            ref_matrix /= np.maximum(
                np.linalg.norm(ref_matrix, axis=1, keepdims=True), 1e-12
            )

            results = {}
            clip_seconds = 0.0
            for index, clip_path in enumerate(clip_paths):
                item_started = time.perf_counter()
                output_path = Path("tri/.tune-embeddings") / f"{len(ref_paths) + index}.npy"
                clip_embeddings = extract_embeddings(
                    clip_path,
                    output_path,
                    InferenceDataset,
                    model,
                    config,
                    device,
                )
                clip_embeddings /= np.maximum(
                    np.linalg.norm(clip_embeddings, axis=1, keepdims=True), 1e-12
                )
                scores = (clip_embeddings @ ref_matrix.T).max(axis=0)
                top_indices = np.argsort(scores)[::-1][:5]
                results[clip_path.name] = {
                    "top5": [
                        [ref_paths[ref_index].stem, float(scores[ref_index])]
                        for ref_index in top_indices
                    ],
                    "secs": round(time.perf_counter() - item_started, 3),
                }
                clip_seconds += time.perf_counter() - item_started

        total_seconds = time.perf_counter() - started
        results["_timing"] = {
            "ref_embedding_seconds": round(ref_seconds, 3),
            "clip_embedding_and_scoring_seconds": round(clip_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }
        with Path("tune.json").open("w") as file:
            json.dump(results, file, ensure_ascii=False)
        print(
            f"clips={len(clip_paths)} refs={len(ref_paths)} "
            f"total_seconds={total_seconds:.3f}"
        )
        return 0
    except Exception as exc:
        import traceback
        total_seconds = time.perf_counter() - started
        print("error", type(exc).__name__, [(f.name, f.lineno) for f in traceback.extract_tb(exc.__traceback__)])
        print(
            f"clips={len(clip_paths)} refs={len(ref_paths)} "
            f"failed=1 total_seconds={total_seconds:.3f}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
