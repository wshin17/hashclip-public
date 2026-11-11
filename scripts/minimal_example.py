"""HashCLIP on the Flickr30k test split: Hamming shortlist, then cosine reranking.

    python scripts/minimal_example.py [--output results.json]
"""
import argparse
import json

import numpy as np

from hashclip.numpy_adapter import load_adapters
from hashclip.retrieve import random_lsh_pack

CHECKPOINTS = {
    0: "runs/2026-11-05_icde-reselect-002/seed0/checkpoints/hashclip_b128_seed0_reselect.numpy.npz",
    1: "runs/2026-11-05_icde-reselect-002/seed1/checkpoints/hashclip_b128_seed1_reselect.numpy.npz",
    42: "runs/2026-11-05_icde-reselect-001/seed42/checkpoints/hashclip_b128_seed42_reselect.numpy.npz",
}
BUDGETS = (50, 100, 200, 500, 1000, 2000, 5000)


def load(path):
    d = np.load(path, allow_pickle=True)
    img = d["img_feat"].astype(np.float32)
    txt = d["txt_feat"].astype(np.float32)
    img /= np.maximum(np.linalg.norm(img, axis=1, keepdims=True), 1e-12)
    txt /= np.maximum(np.linalg.norm(txt, axis=1, keepdims=True), 1e-12)
    return d["img_feat"], d["txt_feat"], img, txt, d["txt_to_img"].astype(np.int64)


def exact_top_k(scores, k):
    ids = np.arange(scores.shape[1])
    out = np.empty((len(scores), k), dtype=np.int64)
    for i, row in enumerate(scores):
        key = -row
        sel = np.flatnonzero(key <= np.partition(key, k - 1)[k - 1])
        out[i] = sel[np.lexsort((ids[sel], key[sel]))[:k]]
    return out


def hamming_order(query_codes, gallery_codes):
    bits = query_codes.shape[1] * 8
    qs = np.unpackbits(query_codes, axis=1).astype(np.float32) * 2 - 1
    gs = np.unpackbits(gallery_codes, axis=1).astype(np.float32) * 2 - 1
    dist = ((bits - qs @ gs.T) / 2).astype(np.int64)
    n = dist.shape[1]
    return np.argsort(dist * n + np.arange(n)[None, :], axis=1, kind="stable")


def recall(final, t2i, direction):
    out = {}
    for k in (1, 5, 10):
        f = final[:, :k]
        if direction == "t2i":
            hits = (f == t2i[:, None]).any(axis=1)
        else:
            hits = (t2i[f] == np.arange(len(f))[:, None]).any(axis=1)
        out[k] = 100 * hits.mean()
    return out


def evaluate(codes, queries, gallery, t2i, direction, reference):
    order = hamming_order(*codes)
    out = {"hash_only": recall(order[:, :10], t2i, direction)}
    for budget in BUDGETS:
        shortlist = order[:, :min(budget, gallery.shape[0])]
        final = np.empty((len(queries), 10), dtype=np.int64)
        for i, cand in enumerate(shortlist):
            scores = gallery[cand] @ queries[i]
            final[i] = cand[np.lexsort((cand, -scores))[:10]]
        retention = np.mean([np.isin(r, c).sum() / 10 for r, c in zip(reference, shortlist)]) * 100
        out[budget] = {"task": recall(final, t2i, direction), "retention": retention}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/backbone_test.npz")
    parser.add_argument("--output")
    args = parser.parse_args()

    raw_img, raw_txt, img, txt, t2i = load(args.features)
    reference = {"t2i": exact_top_k(txt @ img.T, 10), "i2t": exact_top_k(img @ txt.T, 10)}
    result = {"exact": {d: recall(reference[d], t2i, d) for d in reference}, "hashclip": {}, "lsh": {}}
    for seed, path in CHECKPOINTS.items():
        adapters, _ = load_adapters(path)
        codes = {"hashclip": {"img": adapters["img"].encode(raw_img), "txt": adapters["txt"].encode(raw_txt)},
                 "lsh": {"img": random_lsh_pack(img, 128, seed=seed), "txt": random_lsh_pack(txt, 128, seed=seed)}}
        for name, c in codes.items():
            result[name][seed] = {
                "t2i": evaluate((c["txt"], c["img"]), txt, img, t2i, "t2i", reference["t2i"]),
                "i2t": evaluate((c["img"], c["txt"]), img, txt, t2i, "i2t", reference["i2t"]),
            }
            r = result[name][seed]
            print(f"{name} seed {seed}: R@1 t2i/i2t hash-only {r['t2i']['hash_only'][1]:.2f}/"
                  f"{r['i2t']['hash_only'][1]:.2f}, M=500 {r['t2i'][500]['task'][1]:.2f}/"
                  f"{r['i2t'][500]['task'][1]:.2f}")
    print(f"exact: R@1 t2i/i2t {result['exact']['t2i'][1]:.2f}/{result['exact']['i2t'][1]:.2f}")
    if args.output:
        json.dump(result, open(args.output, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
