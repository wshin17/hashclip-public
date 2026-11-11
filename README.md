# HashCLIP

Code and results for "HashCLIP: Binary Candidate Indexing for Image-Text Retrieval
with Frozen Dual Encoders" (ICDE 2027 submission).

## Setup

```bash
uv sync --frozen --extra evaluation                                  # CPU: search and timing
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --frozen --extra encoding   # GPU: features and training
```

## Data

Flickr30k (Karpathy split) and COCO, encoded with the frozen SigLIP model
`hf-hub:timm/ViT-B-16-SigLIP`. Flickr30k is a gated Hugging Face dataset, so run
`huggingface-cli login` first.

```bash
python -m hashclip.cli download-data
python -m hashclip.cli build-splits
python -m hashclip.cli encode-backbone        # then move outputs/features/*.npz to data/features/
python -m hashclip.coco prepare
python -m hashclip.coco encode
python -m hashclip.coco_large download        # expanded COCO gallery
python -m hashclip.coco_large prepare
python -m hashclip.coco_large_encode encode
python -m hashclip.coco_large_encode assemble
```

## Running

Adapters are trained with `python -m hashclip.cli train` and `python -m hashclip.reselect`.
The experiments are modules under `src/hashclip/`, and the configurations used in the
paper are in `configs/`, e.g.

```bash
python -m hashclip.systems_cpu --dataset coco_large --output runs/systems/coco_large
python -m hashclip.opq_timing --config configs/icde_opq.yaml
```

`scripts/minimal_example.py` runs HashCLIP on the Flickr30k test split (Hamming
shortlist, then cosine reranking) and prints Recall@K.

## Results

`results/` contains the CSV/JSON files reported in the paper: timing in
`results/systems/`, retrieval quality in `results/quality/` and the design study in
`results/ablation/`. Timing was measured on an Intel Core i9-14900K with one thread and
Faiss 1.9.0; adapters were trained on one RTX 4090. `results/fig1_image_licences.csv`
lists the licences of the COCO images shown in Fig. 1.

## License

MIT
