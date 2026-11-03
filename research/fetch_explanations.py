"""Fetch Neuronpedia auto-interp explanations for Gemma Scope 16k residual
features (cached on disk; polite rate limit).

usage: python fetch_explanations.py LAYER:IDX [LAYER:IDX ...]
   or  from fetch_explanations import explain; explain(10, 1234)
"""
import json
import os
import sys
import time
import urllib.request

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_final", "np_explanations.json")
URL = "https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-res-16k/{idx}"
_cache = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    return _cache


def save():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(_load(), open(CACHE, "w"), indent=0, sort_keys=True)


def explain(layer: int, idx: int, retries: int = 3) -> dict:
    c = _load()
    key = f"{layer}:{idx}"
    if key in c:
        return c[key]
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(URL.format(layer=layer, idx=idx), timeout=30) as r:
                d = json.load(r)
            expl = [e.get("description", "") for e in d.get("explanations", [])]
            pos = d.get("pos_str") or []
            out = {"explanation": expl[0] if expl else "", "top_logits": pos[:8],
                   "frac_nonzero": d.get("frac_nonzero")}
            c[key] = out
            time.sleep(0.25)
            return out
        except Exception as e:  # network hiccup / rate limit
            time.sleep(2 ** attempt)
            err = str(e)
    return {"explanation": "", "error": err}


if __name__ == "__main__":
    for a in sys.argv[1:]:
        L, i = map(int, a.split(":"))
        print(a, explain(L, i))
    save()
