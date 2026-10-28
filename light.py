import glob, json, os, sys, time
import torch
from transformers import pipeline

model, chunk_dir = sys.argv[1], sys.argv[2]
torch.set_num_threads(os.cpu_count())
t = time.time()
asr = pipeline("automatic-speech-recognition", model=model, device="cpu")
print(f"[{model}] loaded in {time.time() - t:.0f}s on {os.cpu_count()} cpus", flush=True)
out, total = {}, 0.0
for f in sorted(glob.glob(f"{chunk_dir}/c*.wav")):
    t = time.time()
    text = asr(f)["text"]
    took = time.time() - t; total += took
    out[os.path.basename(f)] = {"seconds": round(took, 1), "text": text}
    print(f"[{model}] {os.path.basename(f)} ({took:.0f}s for 30s audio) {text[:200]}", flush=True)
print(f"[{model}] TOTAL {total:.0f}s for {30*len(out)}s audio", flush=True)
json.dump(out, open(f"transcript-{model.replace('/', '_')}.json", "w"), ensure_ascii=False, indent=1)
