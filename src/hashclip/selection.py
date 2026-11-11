"""CPU validation for paired semantic-Recall and candidate-retention selection."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch

from .retrieve import hamming_distance_matrix, prepare_vectors, score_rows, stable_topk


class CandidateValidation:
    def __init__(self, model, features, fractions=(.01, .02, .05, .1), batch=32, semantic_budget=1000):
        self.model = deepcopy(model).cpu().float().eval()
        self.img = np.ascontiguousarray(features.img_feat, dtype=np.float32)
        self.txt = np.ascontiguousarray(features.txt_feat, dtype=np.float32)
        self.mapping = np.asarray(features.txt_to_img, dtype=np.int64)
        self.fractions = tuple(fractions)
        if not self.fractions or any(f <= 0 or f > 1 for f in self.fractions):
            raise ValueError("Candidate fractions must be in (0, 1]")
        self.batch = batch
        self.semantic_budget = semantic_budget
        image, text = prepare_vectors(self.img), prepare_vectors(self.txt)
        self.scores, self.references = {}, {}
        for direction, q, g in (("t2i", text, image), ("i2t", image, text)):
            scores = np.empty((len(q), len(g)), dtype=np.float32)
            reference = np.empty((len(q), 10), dtype=np.int64)
            ids = np.arange(len(g))
            for i, query in enumerate(q):
                scores[i] = score_rows(query, g)
                reference[i] = stable_topk(scores[i], ids, 10, largest=True)
            self.scores[direction], self.references[direction] = scores, reference

    @torch.inference_mode()
    def _encode(self, adapter, features, batch):
        return np.concatenate([np.packbits(adapter(torch.from_numpy(features[start:start + batch])).numpy() > 0, axis=1)
                               for start in range(0, len(features), batch)])

    @torch.inference_mode()
    def __call__(self, model):
        self.model.load_state_dict(model.state_dict())
        img_q = self._encode(self.model.img_adapter, self.img, self.batch)
        txt_q = self._encode(self.model.txt_adapter, self.txt, self.batch)
        img_g = self._encode(self.model.img_adapter, self.img, 256)
        txt_g = self._encode(self.model.txt_adapter, self.txt, 256)
        metrics, selection = {}, []
        for direction, q, g in (("t2i", txt_q, img_g), ("i2t", img_q, txt_g)):
            budgets = [min(len(g), max(10, math.ceil(f * len(g)))) for f in self.fractions]
            width = min(len(g), max(self.semantic_budget, max(budgets)))
            ids = np.arange(len(g))
            recall, complete = np.zeros(len(budgets)), np.zeros(len(budgets))
            semantic = np.zeros(3)
            for i, query in enumerate(q):
                distances = hamming_distance_matrix(query[None], g)[0]
                order = stable_topk(distances, ids, width)
                ref = self.references[direction][i]
                for j, budget in enumerate(budgets):
                    hits = np.isin(ref, order[:budget])
                    recall[j] += hits.mean()
                    complete[j] += hits.all()
                pool = order[:min(self.semantic_budget, len(g))]
                final = stable_topk(self.scores[direction][i, pool], pool, 10, largest=True)
                relevant = (final == self.mapping[i] if direction == "t2i" else self.mapping[final] == i)
                for j, k in enumerate((1, 5, 10)):
                    semantic[j] += relevant[:k].any()
            for fraction, budget, retained, recovered in zip(self.fractions, budgets, recall / len(q), complete / len(q)):
                suffix = f"{direction}_fraction_{fraction:g}"
                metrics[f"candidate_recall_{suffix}"] = float(retained)
                metrics[f"candidate_complete_{suffix}"] = float(recovered)
                metrics[f"candidate_M_{suffix}"] = budget
                selection.append(retained)
            for k, value in zip((1, 5, 10), semantic / len(q) * 100):
                metrics[f"R{k}_{direction}"] = float(value)
        metrics["candidate_selection_score"] = float(np.mean(selection))
        metrics["rsum"] = sum(metrics[f"R{k}_{direction}"] for k in (1, 5, 10) for direction in ("t2i", "i2t"))
        return metrics
