import glob, json, os, sys, time
import torch
from huggingface_hub import hf_hub_download
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

lang = sys.argv[1] if len(sys.argv) > 1 else "yor_Latn"
torch.set_num_threads(os.cpu_count())
t = time.time()
model = load_model("omniASR_LLM_300M", dtype=torch.bfloat16, device=torch.device("cpu"))
model.load_state_dict(torch.load(hf_hub_download("LyngualLabs/yecsASR-omni-llm", "model.pt"), map_location="cpu"), strict=False)
pipe = ASRInferencePipeline(None, model=model, tokenizer=load_tokenizer("omniASR_tokenizer_v1"), device="cpu")
print(f"model loaded in {time.time() - t:.0f}s on {os.cpu_count()} cpus", flush=True)

out = {}
for f in sorted(glob.glob("chunks/c*.wav")):
    t = time.time()
    text = pipe.transcribe([f], lang=[lang])[0]
    took = time.time() - t
    out[os.path.basename(f)] = {"seconds": round(took, 1), "text": text}
    print(f"{os.path.basename(f)} ({took:.0f}s for 30s audio) {text}", flush=True)
json.dump(out, open("transcript.json", "w"), ensure_ascii=False, indent=1)
