import argparse
import copy
import random
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, read_config, write_json
from .model import make_model, model_item, padded_length


def train(cache, output, epochs=None, steps_per_epoch=0, device='cpu', threads=4, config=None):
    torch.set_num_threads(threads)
    manifest = read_json(Path(cache)/'manifest.json'); cfg = copy.deepcopy(manifest['config'])
    if config is not None:
        requested = read_config(config)
        if requested['preprocess'] != cfg['preprocess'] or requested['seed'] != cfg['seed']:
            raise ValueError('Preprocessing or split seed changed; prepare a new cache')
        cfg = requested
    if epochs is not None:
        cfg['train']['epochs'] = epochs
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    records = [r for r in manifest['records'] if r['split']=='train']
    maxlen = padded_length(max(r['tokens'] for r in manifest['records']), cfg)
    if maxlen > cfg['model']['max_tokens']:
        raise ValueError('Padded sequence exceeds max_tokens')
    model = make_model(cfg, len(manifest['feature_names']), maxlen).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'],
                                  weight_decay=cfg['train']['weight_decay'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    history = []; best_key = (-1., float('-inf'))
    for epoch in range(cfg['train']['epochs']):
        model.train(); random.shuffle(records)
        losses = []; batches = [records[i:i+cfg['train']['batch_size']]
                                for i in range(0,len(records),cfg['train']['batch_size'])]
        for step, batch in enumerate(batches):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            items = []
            for r in batch:
                with np.load(str(Path(cache)/r['feature_path'])) as z:
                    x = z['features']
                items.append(model_item(x, r, manifest['normalization'], training=True))
            optimizer.zero_grad()
            loss = model(items)
            if not all(torch.isfinite(v).all() for v in loss.values()):
                raise FloatingPointError('Nonfinite loss: %s' % loss)
            loss['final_loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradients')
            optimizer.step()
            losses.append({k:float(v.detach()) for k,v in loss.items()})
        if not losses:
            raise ValueError('No training batches')
        # Validation only; the held-out test split is never used for checkpoint selection.
        from .evaluate import evaluate_model, checkpoint_selection_key
        metrics = evaluate_model(model, manifest, cache, 'validation', cfg['inference'],
                                 evaluation=cfg.get('evaluation'))
        row = {'epoch':epoch+1, 'loss':{k:float(np.mean([l[k] for l in losses])) for k in losses[0]},
               'validation': metrics}
        history.append(row); print(row, flush=True)
        checkpoint = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch+1,
                      'config':cfg, 'max_seq_len':maxlen, 'input_dim':len(manifest['feature_names']),
                      'feature_names':manifest['feature_names'], 'normalization':manifest['normalization'],
                      'validation_metrics':metrics, 'loss_normalizer':model.loss_normalizer,
                      'debug_subset':manifest['debug_subset'],
                      'selection_rule':'maximum validation F1; ties: minimum matched boundary MAE'}
        torch.save(checkpoint, str(output/'last.pt'))
        selection_key = checkpoint_selection_key(metrics)
        if selection_key > best_key:
            best_key = selection_key; torch.save(checkpoint, str(output/'best.pt'))
        write_json(output/'history.json', history)
    return history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT/'artifacts/intake')
    p.add_argument('--output', type=Path, default=ROOT/'runs/actionformer')
    p.add_argument('--epochs', type=int)
    p.add_argument('--config', type=Path, help='YAML or JSON; defaults to the cached configuration')
    p.add_argument('--steps-per-epoch', type=int, default=0, help='0 = all batches')
    p.add_argument('--device', default='cpu'); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); train(a.cache,a.output,a.epochs,a.steps_per_epoch,a.device,a.threads,a.config)

if __name__=='__main__':main()
