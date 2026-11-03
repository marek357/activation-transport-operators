"""Selective downloader for TheRootOf3/ato-activations.

The dataset stores each layer as separate blosc-compressed zarr chunks inside
a zip, so we can HTTP-range-read only the layers we need instead of pulling
the full ~66 GB. Output: one float16 memmap per layer of shape
[n_seq, 256, d_model] plus attention_mask / input_ids (int32) and the exact
position-0 (BOS) row in float32.
"""
import argparse, json, os, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import fsspec
from numcodecs import Blosc

BASE = ("https://huggingface.co/datasets/TheRootOf3/ato-activations/resolve/main/"
        "activations-gemma2-2b-slimpajama-250k/activations_part_{:04d}.zarr.zip")
N_PARTS = 8
SEQ_PER_PART = 160
D = 2304
T = 256


def open_zip(p, block=2**24):
    fs = fsspec.filesystem("http")
    return zipfile.ZipFile(fs.open(BASE.format(p), "rb", block_size=block))


def part_nseq(p):
    z = open_zip(p, 2**16)
    return json.loads(z.read("activations/layer_0/zarr.json"))["shape"][0]


def decode(raw, shape, dtype):
    return np.frombuffer(Blosc().decode(raw), dtype=dtype).reshape(shape)


def fetch_meta(out, nseqs):
    n = sum(nseqs)
    am = np.zeros((n, T), np.int32); ids = np.zeros((n, T), np.int32)
    off = 0
    for p, ns in enumerate(nseqs):
        z = open_zip(p, 2**16)
        for name, arr in (("attention_mask", am), ("input_ids", ids)):
            meta = json.loads(z.read(f"{name}/zarr.json"))
            dt = np.dtype(meta["data_type"])
            for c in range(2):
                blk = decode(z.read(f"{name}/c/0/{c}"), (ns, 128), dt)
                arr[off:off + ns, c * 128:(c + 1) * 128] = blk
        off += ns
    np.save(os.path.join(out, "attention_mask.npy"), am)
    np.save(os.path.join(out, "input_ids.npy"), ids)


def fetch_layer(out, layer, nseqs, workers=4):
    path = os.path.join(out, f"layer_{layer}.f16")
    done = path + ".done"
    if os.path.exists(done):
        print(f"layer {layer}: already done"); return
    n = sum(nseqs)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, T, D))
    bos = np.zeros((n, D), np.float32)
    offs = np.cumsum([0] + nseqs)
    jobs = [(p, c) for p in range(N_PARTS) for c in range(2)]
    stats = {"maxabs_nonbos": 0.0}

    def job(pc):
        p, c = pc
        for attempt in range(5):
            try:
                z = open_zip(p)
                raw = z.read(f"activations/layer_{layer}/c/0/{c}/0")
                break
            except Exception as e:  # network hiccup -> retry
                print(f"retry {p},{c}: {e}", file=sys.stderr); time.sleep(2 ** attempt)
        ns = nseqs[p]
        blk = decode(raw, (ns, 128, D), np.float32)
        if c == 0:
            bos[offs[p]:offs[p + 1]] = blk[:, 0]
            m = np.abs(blk[:, 1:]).max()
        else:
            m = np.abs(blk).max()
        stats["maxabs_nonbos"] = max(stats["maxabs_nonbos"], float(m))
        mm[offs[p]:offs[p + 1], c * 128:(c + 1) * 128] = blk.astype(np.float16)
        return len(raw)

    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        tot = sum(ex.map(job, jobs))
    mm.flush(); del mm
    np.save(os.path.join(out, f"layer_{layer}_bos.npy"), bos)
    if stats["maxabs_nonbos"] > 60000:
        print(f"WARNING layer {layer}: max |x| {stats['maxabs_nonbos']} overflows fp16")
    open(done, "w").write(json.dumps(stats))
    print(f"layer {layer}: {tot/1e9:.2f} GB in {time.time()-t:.0f}s, max|x| non-BOS={stats['maxabs_nonbos']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/user/ato_data")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nseqs = [part_nseq(p) for p in range(N_PARTS)]
    print("seqs per part", nseqs)
    if not os.path.exists(os.path.join(a.out, "input_ids.npy")):
        fetch_meta(a.out, nseqs)
    for L in a.layers:
        fetch_layer(a.out, L, nseqs, a.workers)
