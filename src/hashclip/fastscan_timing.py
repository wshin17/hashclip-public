"""D-012 sequential fitting, native FastScan evaluation and same-domain anchors."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

from .opq_timing import check_identity, read_json, run_dataset, save_json
from .systems_cpu import resolve_path
from .paths import ROOT


def fit_models(config):
    for options in config['datasets'].values():
        features = resolve_path(options['training_features'])
        expected = read_json(resolve_path(options['training_source']))['training_features']
        check_identity(features,expected)
        output = resolve_path(options['models'])
        if (output/'manifest.json').exists():
            manifest = read_json(output/'manifest.json')
            if {(r['modality'],r['seed']) for r in manifest['models'] if r['method']=='fastscan'} == {
                    (m,s) for m in ('img','txt') for s in (0,1,42)}:
                for row in manifest['models']:
                    check_identity(output/Path(row['artifact']['path']).name,row['artifact'])
                continue
        if time.time() >= datetime.fromisoformat(config['deadline']).timestamp():
            raise TimeoutError('D-012 deadline reached; preserve partial results')
        command = ['taskset','-c','16-31',sys.executable,'-m','hashclip.train_representations',
                   '--features',str(features),'--output',str(output),'--threads','8',
                   '--methods','fastscan','--seeds','0','1','42','--resume']
        subprocess.run(command,check=True)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/icde_fastscan.yaml')
    parser.add_argument('--fit-only',action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    fit_models(config)
    if args.fit_only:
        return
    output = resolve_path(config['output'])
    results = [run_dataset(name,options,config,output) for name,options in config['datasets'].items()]
    save_json(output/'fastscan_complete.json',{'datasets':results,'completed_utc':time.time()})


if __name__ == '__main__':
    main()
