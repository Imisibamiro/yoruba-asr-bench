import audioop, glob, json, os, sys, time, wave
import torch
from transformers import pipeline

shard, shards = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(os.cpu_count())
pieces = [p for i, p in enumerate(sorted(glob.glob("pieces/*.wav"))) if i % shards == shard]
def loud(f):
    w = wave.open(f); return audioop.rms(w.readframes(w.getnframes()), 2)
lid = pipeline("audio-classification", model="facebook/mms-lid-126", device="cpu")
asr = {"yor": pipeline("automatic-speech-recognition", model="NCAIR1/Yoruba-ASR", device="cpu"),
       "ibo": pipeline("automatic-speech-recognition", model="NCAIR1/Igbo-ASR", device="cpu")}
gen = dict(no_repeat_ngram_size=3, repetition_penalty=1.3, num_beams=3)
out, t = {}, time.time()
for f in pieces:
    rec = {"rms": loud(f)}
    if rec["rms"] >= 300:
        top = lid(f, top_k=3); rec["lid"] = [(x["label"], round(x["score"], 2)) for x in top]
        lang, score = top[0]["label"], top[0]["score"]
        if lang in asr and score >= 0.4:
            rec["lang"] = lang; rec["text"] = asr[lang](f, generate_kwargs=gen)["text"]
    out[os.path.basename(f)] = rec
print(f"shard {shard}: {len(pieces)} pieces in {time.time() - t:.0f}s; transcribed {sum('text' in r for r in out.values())}", flush=True)
json.dump(out, open(f"out-{shard}.json", "w"), ensure_ascii=False)
