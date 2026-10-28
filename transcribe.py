import glob, json, os, sys, time
import torch
from huggingface_hub import hf_hub_download
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

which, lang, chunk_dir = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(os.cpu_count())
t = time.time()
if which == "llm-yoruba":
    model = load_model("omniASR_LLM_300M", dtype=torch.bfloat16, device=torch.device("cpu"))
    model.load_state_dict(torch.load(hf_hub_download("LyngualLabs/yecsASR-omni-llm", "model.pt"), map_location="cpu"), strict=False)
    pipe = ASRInferencePipeline(None, model=model, tokenizer=load_tokenizer("omniASR_tokenizer_v1"), device="cpu")
else:
    pipe = ASRInferencePipeline(model_card=which, device="cpu", dtype=torch.float32)
print(f"[{which}] loaded in {time.time() - t:.0f}s on {os.cpu_count()} cpus", flush=True)

out, total = {}, 0.0
for f in sorted(glob.glob(f"{chunk_dir}/c*.wav")):
    t = time.time()
    text = pipe.transcribe([f], lang=[lang])[0]
    took = time.time() - t; total += took
    out[os.path.basename(f)] = {"seconds": round(took, 1), "text": text}
    print(f"[{which}] {os.path.basename(f)} ({took:.0f}s for 30s audio) {text[:200]}", flush=True)
print(f"[{which}] TOTAL {total:.0f}s for {30*len(out)}s audio", flush=True)
json.dump(out, open(f"transcript-{which}.json", "w"), ensure_ascii=False, indent=1)
