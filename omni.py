import glob, json, os, sys, time
import torch
which = sys.argv[1]
torch.set_num_threads(os.cpu_count())
PROMPT = ("This is 90 seconds of a Nigerian church service livestream. Transcribe ONLY what is being SUNG, word for word, "
          "in its original language (Yoruba, Igbo, Pidgin or English). Do not translate or summarise; write [unclear] where you "
          "cannot hear. If there is no singing, say NO SINGING. Last line: LANGUAGE: <language> | SONG: <title if confident, else unknown>.")
files = sorted(glob.glob("p6/*.wav"))
out = {}
t0 = time.time()
if which == "qwen":
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    name = "Qwen/Qwen2.5-Omni-3B"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cpu")
    if hasattr(model, 'talker'): model.disable_talker()
    proc = Qwen2_5OmniProcessor.from_pretrained(name)
    sysmsg = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    for f in files:
        conv = [{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
                {"role": "user", "content": [{"type": "audio", "audio": f}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        t = time.time()
        ids = model.thinker.generate(**inp, max_new_tokens=400, do_sample=False)
        ids = ids[0] if isinstance(ids, tuple) else ids
        res = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out[os.path.basename(f)] = {"secs": round(time.time() - t), "text": res}
        print(f"[qwen] {os.path.basename(f)} {time.time() - t:.0f}s", flush=True)
else:
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    name = "google/gemma-3n-E2B-it"
    model = Gemma3nForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cpu")
    proc = AutoProcessor.from_pretrained(name)
    for f in files:
        msgs = [{"role": "user", "content": [{"type": "audio", "audio": f}, {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        t = time.time()
        ids = model.generate(**inp, max_new_tokens=400, do_sample=False)
        res = proc.decode(ids[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        out[os.path.basename(f)] = {"secs": round(time.time() - t), "text": res}
        print(f"[gemma] {os.path.basename(f)} {time.time() - t:.0f}s", flush=True)
print(f"[{which}] total {time.time() - t0:.0f}s", flush=True)
json.dump(out, open(f"omni-{which}.json", "w"), ensure_ascii=False)
