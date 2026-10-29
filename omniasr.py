import sys, os, json, time, glob, tempfile
import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

MODEL = sys.argv[1]  # ctc300m or llm300m
CARDS = {"ctc300m": "omniASR_CTC_300M", "llm300m": "omniASR_LLM_300M"}
PIECE = 30
t0 = time.time()
pipe = ASRInferencePipeline(model_card=CARDS[MODEL])
load_s = round(time.time() - t0, 1)
use_lang = MODEL != "ctc300m"
out, n_pieces, tmp = {}, 0, tempfile.mkdtemp()
t1 = time.time()
for path in sorted(glob.glob("tri/clips/*.wav")):
    name = os.path.splitext(os.path.basename(path))[0]
    audio, sr = sf.read(path)
    step = PIECE * sr
    files = []
    for i, s in enumerate(range(0, len(audio), step)):
        piece = audio[s:s + step]
        if len(piece) < sr:
            continue
        f = os.path.join(tmp, f"p{len(files)}.wav")
        sf.write(f, piece, sr)
        files.append(f)
    n_pieces += len(files)
    rec = {"secs": round(len(audio) / sr, 1)}
    if use_lang:
        for key, code in (("yor", "yor_Latn"), ("ibo", "ibo_Latn")):
            rec[key] = " ".join(pipe.transcribe(files, lang=[code] * len(files), batch_size=1))
    else:
        text = " ".join(pipe.transcribe(files, batch_size=1))
        rec["yor"] = text
        rec["ibo"] = text
    out[name] = rec
out["_timing"] = {"load_s": load_s, "transcribe_s": round(time.time() - t1, 1), "clips": len(out), "pieces": n_pieces, "lang_conditioned": use_lang}
json.dump(out, open(f"omniasr-{MODEL}.json", "w"), ensure_ascii=False)
print(f"clips={len(out) - 1} pieces={n_pieces} load_s={load_s} transcribe_s={out['_timing']['transcribe_s']}")
