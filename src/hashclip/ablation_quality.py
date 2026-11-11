"""Re-evaluate the recovered WSDM ablations at fixed tight candidate budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .evaluate_cpu import configure, input_identity, load_features, make_index, write_csv
from .paths import ROOT, runs_root, data_root


def checkpoints():
    return sorted((runs_root() / 'wsdm-original/checkpoints').glob('*seed42*.pt'))


def run(output):
    configure({'threads': 1, 'blas_core': 'Haswell'})
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .numpy_adapter import load_adapters, numpy_checkpoint_path
    from .representation_quality import measure_positions, position_metrics

    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new ablation output directory')
    output.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1)
    root = data_root() / 'features'
    # Resolve the canonical original-gallery references from the actual
    # representation run, rather than selecting new neighbors or scorers.
    original = json.loads((ROOT / 'outputs/tables/icde_representations/flickr_small/run.json').read_text())
    refs = {Path(p['path']).name: ROOT / p['path'] for p in original['inputs']
            if Path(p['path']).name in ('val_reference.npz', 'test_reference.npz')}
    if len(refs) != 2:
        raise ValueError('Both original canonical reference caches are required')
    cfg = {'threads': 1, 'binary_use_heap': False, 'rerank_block': 256,
           'rerank_mode': 'indexed', 'adapter_backend': 'numpy'}
    manifest = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'quality_only': True, 'batch': 32, 'seed': 42,
                'checkpoint_selection': 'unchanged original WSDM saved checkpoints',
                'budget_selection': 'ceil of 1%, 2%, 5%, 10% validation gallery size; same M on test',
                'training': 'none', 'inputs': [], 'checkpoints': [], 'variants': {}}
    models = checkpoints()
    for checkpoint in models:
        _, meta = load_adapters(numpy_checkpoint_path(checkpoint))
        manifest['checkpoints'].append(input_identity(numpy_checkpoint_path(checkpoint)))
        manifest['variants'][checkpoint.stem] = meta
    rows, budgets = [], {}
    for split in ('val', 'test'):
        features = root / f'backbone_{split}.npz'
        arrays = load_features(features)
        reference_path = refs[f'{split}_reference.npz']
        manifest['inputs'].extend(input_identity(p) for p in (features, reference_path))
        (output/'config.json').write_text(json.dumps(manifest, indent=2)+'\n')
        with np.load(reference_path) as cache:
            references = {d: cache[d] for d in ('t2i','i2t')}
        with threadpool_limits(limits=1):
            for direction, qm, gm in (('t2i','txt','img'), ('i2t','img','txt')):
                query, gallery = arrays[qm+'_feat'], arrays[gm+'_feat']
                if split == 'val':
                    budgets[direction] = [max(10, int(np.ceil(f*len(gallery)))) for f in (.01,.02,.05,.1)]
                for checkpoint in models:
                    meta = manifest['variants'][checkpoint.stem]
                    print(split, direction, checkpoint.stem, flush=True)
                    index = make_index({'method':'hashclip','seed':42,'direction':direction,
                                        'checkpoint':str(checkpoint)}, gallery, {**cfg,'bits':meta['bits']})
                    positions = measure_positions(index, query, references[direction], batch=32)
                    np.save(output/f'{split}_{direction}_{checkpoint.stem}_positions.npy', positions)
                    for fraction, budget in zip((.01,.02,.05,.1), budgets[direction]):
                        rows.append({'split':split,'direction':direction,'checkpoint':checkpoint.stem,
                                     'seed':42,'bits':meta['bits'],'hidden_dim':meta['hidden_dim'],
                                     'epoch':meta['epoch'],'validation_gallery_fraction':fraction,
                                     'M':budget,'gallery_size':len(gallery),'queries':len(query),
                                     **position_metrics(positions,budget)})
                    write_csv(output/'curves.csv',rows)
                    del index
    (output/'complete.json').write_text(json.dumps({'records':len(rows),'checkpoints':len(models)},indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.export:
        configure({'threads':1, 'blas_core':'Haswell'})
        from .numpy_adapter import export_checkpoint
        for checkpoint in checkpoints():
            _, _, destination = export_checkpoint(checkpoint)
            print(destination, flush=True)
    elif a.output is not None:
        run(a.output)
    else:
        p.error('--output is required for quality evaluation')


if __name__ == '__main__':
    main()
