"""Event F1 at strict IoU > 0.25 and boundary MAE on matched events."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, read_config, write_json
from .model import predict_run, deduplicate, load_checkpoint


def event_metrics(predictions, targets, tiou=.25):
    """Pool event counts across people; confidence-ordered one-to-one matching.

    Intervals use absolute milliseconds. Equality at the IoU threshold is a miss.
    MAE is in seconds and only includes true-positive matched pairs.
    """
    if not 0 <= tiou < 1:
        raise ValueError('IoU threshold must be in [0, 1)')
    tp=0; fp=0; fn=0; start_errors=[]; end_errors=[]
    for subject in sorted(set(predictions)|set(targets)):
        gt=sorted(set(targets.get(subject, []))); used=set()
        for a,b,score in sorted(predictions.get(subject,[]),key=lambda x:-x[2]):
            candidates=[]
            for i,(x,y) in enumerate(gt):
                if i not in used:
                    inter=max(0,min(b,y)-max(a,x)); union=(b-a)+(y-x)-inter
                    candidates.append((inter/max(1,union),i,x,y))
            best=max(candidates, default=(0,-1,0,0))
            if best[1] >= 0 and best[0] > tiou:
                used.add(best[1]);tp+=1
                start_errors.append(abs(a-best[2])/1000.)
                end_errors.append(abs(b-best[3])/1000.)
            else:fp+=1
        fn+=len(gt)-len(used)
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    start_mae=float(np.mean(start_errors)) if tp else None
    end_mae=float(np.mean(end_errors)) if tp else None
    return {'sensitivity':recall, 'positive_predictive_value':precision,
            'f1':2*tp/max(1,2*tp+fp+fn), 'tp':tp,'fp':fp,'fn':fn,
            'start_mae_seconds':start_mae, 'end_mae_seconds':end_mae,
            'boundary_mae_seconds':(start_mae+end_mae)/2 if tp else None,
            'iou_threshold':tiou, 'iou_comparison':'>'}


def checkpoint_selection_key(metrics):
    """Maximize validation F1; break equal-F1 ties with lower boundary MAE."""
    mae=metrics['boundary_mae_seconds']
    return metrics['f1'], -mae if mae is not None else float('-inf')


def merge_targets(intervals):
    merged=[]
    for a,b in sorted(set(tuple(x) for x in intervals)):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return [tuple(x) for x in merged]


def evaluate_model(model, manifest, cache, split, inference, return_predictions=False, evaluation=None):
    model.eval(); predictions=defaultdict(list);targets=defaultdict(list)
    for r in manifest['records']:
        if r['split']!=split:continue
        with np.load(str(Path(cache)/r['feature_path'])) as z:x=z['features']
        predictions[r['subject']]+=predict_run(model,x,r,manifest['normalization'],
                                               inference['score_threshold'],inference['min_duration_ms'])
        targets[r['subject']]+=r['segments_ms']
    predictions={k:deduplicate(v,inference['nms_iou']) for k,v in predictions.items()}
    targets={k:merge_targets(v) for k,v in targets.items()}
    metrics=event_metrics(predictions,targets,tiou=(evaluation or {}).get('iou_threshold',.25))
    return (metrics,predictions) if return_predictions else metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'artifacts/intake')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/actionformer/best.pt')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--output',type=Path,default=ROOT/'runs/actionformer/evaluation.json')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--config',type=Path,help='Optional YAML/JSON override for evaluation settings only')
    a=p.parse_args();torch.set_num_threads(a.threads)
    manifest=read_json(a.cache/'manifest.json');model,ckpt=load_checkpoint(a.checkpoint,a.device)
    if ckpt['feature_names']!=manifest['feature_names'] or ckpt['normalization']!=manifest['normalization']:
        raise ValueError('Checkpoint/cache feature or normalization mismatch')
    evaluation=(read_config(a.config) if a.config else ckpt['config']).get('evaluation',{})
    metrics,predictions=evaluate_model(model,manifest,a.cache,a.split,ckpt['config']['inference'],True,
                                       evaluation=evaluation)
    write_json(a.output,{'split':a.split,'metrics':metrics,'predictions':predictions,
                         'debug_subset':manifest['debug_subset']});print(metrics)

if __name__=='__main__':main()
