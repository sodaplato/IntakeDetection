"""Build cached feature sequences and eating intervals without modifying raw data."""
import argparse
import hashlib
import datetime
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import ROOT, read_csv, read_json, read_config, write_json, subject_id
from .features import extract_runs, feature_names


def prepare(data_dir, output, config, limit_files=0):
    output = Path(output)
    if (output / 'manifest.json').exists():
        raise ValueError('Cache exists; choose a new --output to prevent stale/mixed runs')
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_config(config)
    exclusions = read_json(Path(data_dir) / 'excluded_subjects.json')
    bad = {subject_id(v) for v in exclusions['excluded_externalids']}
    sensors, meals = read_csv(Path(data_dir) / 'IMU+PPG.csv'), read_csv(Path(data_dir) / 'MealInfo.csv')
    # Group connected account IDs AND participant IDs to avoid aliases crossing splits.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in sensors + meals:
        h, e = row.get('healthid', '').strip(), subject_id(row.get('externalid', ''))
        if h and e:
            parent[find('e:' + e)] = find('h:' + h)
    label_audit = [{'externalid': r['externalid'], 'healthid': r['healthid'],
                    'timeFrame.startTime': r['timeFrame.startTime'], 'timeFrame.endTime': r['timeFrame.endTime'],
                    'beforeTime': r['beforeTime'], 'afterTime': r['afterTime']}
                   for r in meals if (r['timeFrame.startTime'],r['timeFrame.endTime']) != (r['beforeTime'],r['afterTime'])]
    write_json(output/'label_field_disagreements.json', label_audit)
    intervals = defaultdict(list)
    for row in meals:
        e, h = subject_id(row['externalid']), row['healthid'].strip()
        if e in bad:
            continue
        a, b = int(row['timeFrame.startTime']), int(row['timeFrame.endTime'])
        if b > a:
            intervals[(h, e)].append((a, b))
    lookup = defaultdict(list)
    for p in (Path(data_dir) / 'IMU+PPG').rglob('sensorData-*'):
        if p.is_dir():
            lookup[p.name].append(p)
    eligible, skipped = [], []
    for row in sensors:
        h, e = row['healthid'].strip(), subject_id(row['externalid'])
        key = Path(row['sensorData']).stem
        reason = None
        if e in bad:
            reason = 'excluded_subject'
        elif not re.fullmatch(r'HNU\d+[A-Z]?', e):
            reason = 'non_participant_id'
        elif (h, e) not in intervals:
            reason = 'no_meal_annotations_for_account_and_id'
        else:
            folders = lookup[key]
            if len(folders) != 1 or len(list(folders[0].glob('*.txt'))) != 1:
                raise ValueError('Missing/ambiguous sensor attachment: ' + key)
            txt = next(folders[0].glob('*.txt'))
            with txt.open('rb') as stream:
                header = stream.read(16)
            if not header.lstrip(b'\xef\xbb\xbf').startswith(b'ACC_TIME\t'):
                skipped.append({'id': key, 'reason': 'unsupported_non_tabular_txt', 'source': str(txt)})
                continue
            zone = datetime.timezone(datetime.timedelta(hours=8))
            start = datetime.datetime.strptime(row['timeStamp.startTime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=zone).timestamp()*1000
            end = datetime.datetime.strptime(row['timeStamp.endTime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=zone).timestamp()*1000
            overlap = any(a < end and b > start for a,b in intervals[(h,e)])
            eligible.append({'id': key, 'path': txt, 'healthid': h,
                             'subject': e, 'group': find('h:' + h), 'metadata_positive': overlap})
        if reason:
            skipped.append({'id': key, 'reason': reason})
    # Debug subset prioritizes short positive files from different groups to exercise regression.
    # Not a representative evaluation set; manifest records debug_subset=True.
    if limit_files:
        ordered = sorted(eligible, key=lambda x: (not x['metadata_positive'], x['path'].stat().st_size, x['id']))
        selected, seen = [], set()
        for item in ordered:
            if item['group'] not in seen:
                selected.append(item); seen.add(item['group'])
            if len(selected) == limit_files:
                break
        if len(selected) < limit_files:
            selected += [x for x in ordered if x not in selected][:limit_files-len(selected)]
        eligible = selected
    groups = sorted({x['group'] for x in eligible},
                    key=lambda x: hashlib.sha256((str(cfg['seed']) + x).encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError('Need >=3 independent account/participant groups for train/validation/test')
    nval = max(1, int(len(groups) * .15)); ntest = max(1, int(len(groups) * .15))
    split = {g: ('validation' if i < nval else 'test' if i < nval+ntest else 'train') for i, g in enumerate(groups)}
    records, rejected = [], []
    for idx, item in enumerate(eligible):
        print('Extract %d/%d %s %s' % (idx+1, len(eligible), item['subject'], item['id']), flush=True)
        runs = extract_runs(item['path'], cfg['preprocess'])
        for run_idx, run in enumerate(runs):
            tokens = run['features'].shape[1]
            if tokens > cfg['model']['max_tokens']:
                rejected.append({'id': item['id'], 'run': run_idx, 'tokens': tokens,
                                 'reason': 'exceeds_max_tokens_raise_config_to_preserve_full_run'})
                continue  # Never silently truncate an entire meal or resize the time axis.
            a, b = run['valid_start_ms'], run['valid_end_ms']
            segments = sorted(set((max(a, x), min(b, y)) for x, y in intervals[(item['healthid'], item['subject'])]
                                  if x < b and y > a))
            key = item['id'] + '_run%d' % run_idx
            dest = output / 'features' / (key + '.npz')
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(str(dest), features=run['features'])
            records.append({'id': key, 'file_id': item['id'], 'source': str(item['path'].resolve()),
                            'subject': item['subject'], 'healthid': item['healthid'], 'group': item['group'],
                            'split': split[item['group']], 'feature_path': str(dest.relative_to(output)),
                            'tokens': tokens, 'segments_ms': segments,
                            **{k: v for k, v in run.items() if k != 'features'}})
    used = {r['split'] for r in records}
    if used != {'train', 'validation', 'test'}:
        raise ValueError('Not all splits contain valid runs; increase --limit-files or check rejected records')
    # Fit per-channel normalization on training only, never on validation/test.
    sums = np.zeros(len(feature_names(cfg['preprocess'])), np.float64)
    sums2 = sums.copy(); count = 0
    for r in records:
        if r['split'] == 'train':
            with np.load(str(output / r['feature_path'])) as z:
                x = z['features'].astype(np.float64)
            sums += x.sum(axis=1); sums2 += (x*x).sum(axis=1); count += x.shape[1]
    mean = sums/count; std = np.maximum(np.sqrt(np.maximum(sums2/count-mean**2, 0)), .01)
    norm = {'mean': mean.tolist(), 'std': std.tolist(), 'training_tokens': count}
    summary = {s: {'runs': sum(r['split']==s for r in records),
                   'positive_runs': sum(r['split']==s and bool(r['segments_ms']) for r in records),
                   'subjects': sorted({r['subject'] for r in records if r['split']==s})}
               for s in ['train','validation','test']}
    manifest = {'config': cfg, 'feature_names': feature_names(cfg['preprocess']), 'normalization': norm,
                'records': records, 'skipped_indexes': skipped, 'rejected_runs': rejected,
                'summary': summary, 'debug_subset': bool(limit_files),
                'preparation_limit_files': limit_files,
                'background_assumption': 'Outside logged meal intervals treated as background; completeness unverified.',
                'label_join': 'exact healthid AND normalized externalid',
                'label_source': ['timeFrame.startTime', 'timeFrame.endTime'],
                'label_field_disagreement_count': len(label_audit)}
    write_json(output / 'manifest.json', manifest)
    print(summary, flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=ROOT/'Data')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/intake')
    p.add_argument('--config', type=Path, default=ROOT/'experiment261016/intake_actionformer.yaml')
    p.add_argument('--limit-files', type=int, default=0, help='Debug only; 0 processes all eligible files')
    a = p.parse_args(); prepare(a.data_dir, a.output, a.config, a.limit_files)

if __name__ == '__main__':
    main()
