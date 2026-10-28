import glob, json, os, sys, time
import torch
from transformers import pipeline

torch.set_num_threads(os.cpu_count())
gen = dict(no_repeat_ngram_size=3, repetition_penalty=1.3, num_beams=3)
pieces = sorted(glob.glob("pieces/*.wav"))
out = {}
for model in ("NCAIR1/Yoruba-ASR", "NCAIR1/Igbo-ASR"):
    asr = pipeline("automatic-speech-recognition", model=model, device="cpu")
    t = time.time()
    for f in pieces:
        out.setdefault(os.path.basename(f), {})[model] = asr(f, generate_kwargs=gen)["text"]
    print(f"[{model}] {len(pieces)} pieces in {time.time() - t:.0f}s", flush=True)
json.dump(out, open("out.json", "w"), ensure_ascii=False)
