"""Single-node job: prepare/reuse compatible features, train, then held-out test."""
import argparse
import os
from pathlib import Path
import torch
from .common import ROOT, read_config, read_json, write_json
from .prepare import prepare
from .train import train
from .evaluate import evaluate_model
from .model import load_checkpoint


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def check_cache(manifest, config, limit_files):
    old = manifest['config']
    if old['seed'] != config['seed'] or old['preprocess'] != config['preprocess']:
        raise ValueError('Cached preprocessing/split differs; change run.cache_dir')
    if old['model']['max_tokens'] != config['model']['max_tokens']:
        raise ValueError('Cached sequence limit differs; change run.cache_dir')
    if manifest.get('debug_subset', False) != bool(limit_files):
        raise ValueError('Cannot mix full-data and debug caches; change run.cache_dir')
    previous_limit = manifest.get('preparation_limit_files')
    if previous_limit is None:
        previous_limit = len({r['file_id'] for r in manifest['records']}) if manifest.get('debug_subset') else 0
    if previous_limit != limit_files:
        raise ValueError('Cached debug file limit differs; change run.cache_dir')
    if manifest.get('rejected_runs'):
        raise ValueError('Some runs exceed max_tokens; review rejected_runs and rebuild with a larger limit')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'experiment261016/intake_actionformer.yaml')
    p.add_argument('--run-id', default=os.environ.get('SLURM_JOB_ID'))
    p.add_argument('--threads', type=int)
    a = p.parse_args()
    cfg = read_config(a.config)
    run = cfg['run']
    device = run['device']
    threads = a.threads or run['threads']
    torch.set_num_threads(threads)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('YAML requests CUDA but no CUDA device is available in this job')
    data, cache, output = [project_path(run[k]) for k in ('data_dir','cache_dir','output_dir')]
    if a.run_id:
        if not a.run_id.replace('_','').replace('-','').isalnum():
            raise ValueError('Invalid run ID')
        output = output/('job_'+a.run_id)
    if (output/'last.pt').exists():
        raise ValueError('Output already contains a checkpoint; select a fresh run ID/output_dir')
    limit = run.get('limit_files', 0)
    cache.mkdir(parents=True, exist_ok=True)
    # Slurm jobs sharing this cache must not prepare it concurrently.
    import fcntl
    with (cache/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = cache/'manifest.json'
        if not manifest_path.exists():
            prepare(data, cache, a.config, limit)
        manifest = read_json(manifest_path)
        check_cache(manifest, cfg, limit)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'job_config.json', cfg)
    print('Config: %s; cache: %s; output: %s; device: %s' % (a.config,cache,output,device),flush=True)
    train(cache, output, steps_per_epoch=run.get('steps_per_epoch',0), device=device,
          threads=threads, config=a.config)
    if run.get('evaluate_test', True):
        model, checkpoint = load_checkpoint(output/'best.pt', device)
        metrics, predictions = evaluate_model(model, manifest, cache, 'test', cfg['inference'], True,
                                              evaluation=cfg.get('evaluation'))
        write_json(output/'test_metrics.json', {'split':'test','metrics':metrics,
                   'predictions':predictions,'debug_subset':manifest['debug_subset']})
        print('Test:', metrics, flush=True)
    print('Job completed: %s' % output, flush=True)


if __name__ == '__main__':
    main()
