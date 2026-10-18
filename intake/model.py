import copy
import importlib
import importlib.machinery
import sys
import types
import numpy as np
import torch
from .common import ROOT

# Use the original LocPointTransformer, not a substitute classifier.
sys.path.insert(0, str(ROOT/'Actionformer'))
from libs.core.config import DEFAULTS
from libs.modeling import make_meta_arch


def tridet_modules():
    # Both upstream repositories name their package "libs". Isolate TriDet's
    # registries so model/checkpoint loading never depends on import order.
    namespace = '_intake_tridet_vendor'
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(ROOT/'TriDet-master'/'libs')]
        package.__package__ = namespace
        package.__spec__ = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[namespace] = package
    defaults = importlib.import_module(namespace+'.core.config').DEFAULTS
    modeling = importlib.import_module(namespace+'.modeling')
    return defaults, modeling.make_meta_arch


def make_model(config, input_dim, max_seq_len):
    m = config['model']; levels = m['levels']
    name = m.get('name', 'actionformer').lower()
    if name not in ('actionformer', 'tridet'):
        raise ValueError('Unknown model.name: '+name)
    defaults, factory = tridet_modules() if name == 'tridet' else (DEFAULTS, make_meta_arch)
    model_cfg = copy.deepcopy(defaults['model'])
    train_cfg = copy.deepcopy(defaults['train_cfg'])
    test_cfg = copy.deepcopy(defaults['test_cfg'])
    train_cfg.update(init_loss_norm=20, droppath=.1)
    infer = config['inference']
    test_cfg.update(nms_method='hard', iou_threshold=infer['nms_iou'],
                    min_score=infer['score_threshold'], pre_nms_thresh=.001,
                    pre_nms_topk=1000, max_seg_num=infer['max_segments'], voting_thresh=0)
    ranges = [[0, 4]]
    for i in range(1, levels):
        ranges.append([2**(i+1), 2**(i+2) if i < levels-1 else 1e8])
    model_cfg.update(input_dim=input_dim, num_classes=1, max_seq_len=max_seq_len,
                     backbone_arch=(2, 2, levels-1), regression_range=ranges,
                     embd_dim=m['width'], fpn_dim=m['width'], head_dim=m['width'],
                     max_buffer_len_factor=4, train_cfg=train_cfg, test_cfg=test_cfg)
    if name == 'tridet':
        if m['width'] % 16:
            raise ValueError('TriDet model.width must be divisible by 16 (SGP GroupNorm)')
        model_cfg.update(backbone_type='SGP', n_sgp_win_size=m['sgp_window'],
                         sgp_mlp_dim=m['sgp_mlp_dim'], num_bins=m['num_bins'],
                         k=m['k'], iou_weight_power=m['iou_weight_power'],
                         use_trident_head=True, input_noise=m.get('input_noise',0.))
        return factory('TriDet', **model_cfg)
    model_cfg.update(n_head=m['heads'], n_mha_win_size=m['attention_window'])
    return factory('LocPointTransformer', **model_cfg)


def required_padding(config):
    m = config['model']
    if m.get('name', 'actionformer').lower() == 'tridet':
        window = m['sgp_window']
        if window < 1 or window % 2 != 1:
            raise ValueError('TriDet sgp_window must be a positive odd integer')
        return 2**(m['levels']-1) * window
    return 2**(m['levels']-1) * (m['attention_window']//2*2 if m['attention_window']>1 else 1)


def padded_length(length, config):
    stride = required_padding(config)
    return max(stride, (length+stride-1)//stride*stride)


def model_item(features, run, normalization, training=False):
    mean = np.asarray(normalization['mean'], np.float32)[:, None]
    std = np.asarray(normalization['std'], np.float32)[:, None]
    x = np.clip((features - mean)/std, -20, 20).astype(np.float32)
    item = {'video_id': run.get('id', 'inference'), 'feats': torch.from_numpy(x),
            'fps': 1000.0/run['bin_ms'], 'feat_stride': 1., 'feat_num_frames': 1.,
            'duration': features.shape[1]*run['bin_ms']/1000.}
    if training:
        # Features represent bin centers. This inverses upstream postprocessing's +0.5.
        segments = np.asarray(run['segments_ms'], np.float64).reshape(-1, 2)
        segments = (segments-run['grid_start_ms'])/run['bin_ms'] - .5
        item['segments'] = torch.tensor(segments, dtype=torch.float32)
        item['labels'] = torch.zeros(len(segments), dtype=torch.long)
    return item


def load_checkpoint(path, device='cpu'):
    # Only load trusted project checkpoints. weights_only=False supports stored config.
    try:
        ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location='cpu')
    model = make_model(ckpt['config'], ckpt['input_dim'], ckpt['max_seq_len'])
    model.load_state_dict(ckpt['model']); model.to(device).eval()
    return model, ckpt


def predict_run(model, features, run, normalization, threshold, min_duration_ms):
    # Extend only coordinate buffers, never the learned weights or the time axis.
    needed = (features.shape[1] + model.max_div_factor - 1)//model.max_div_factor*model.max_div_factor
    if needed > model.point_generator.max_seq_len:
        model.point_generator.max_seq_len = needed
        model.point_generator.buffer_points = model.point_generator._generate_points().to(model.device)
    with torch.no_grad():
        result = model([model_item(features, run, normalization)])[0]
    out = []
    for segment, score in zip(result['segments'].tolist(), result['scores'].tolist()):
        if score < threshold:
            continue
        start = max(run['valid_start_ms'], run['grid_start_ms'] + int(round(segment[0]*1000)))
        end = min(run['valid_end_ms'], run['grid_start_ms'] + int(round(segment[1]*1000)))
        if end > start and end-start >= min_duration_ms:
            out.append((start, end, float(score)))
    return out


def deduplicate(intervals, threshold=.5):
    # Apply cross-file NMS on the same participant's absolute time axis.
    kept = []
    for a, b, score in sorted(intervals, key=lambda x: -x[2]):
        if all(max(0, min(b,y)-max(a,x))/max(1, max(b,y)-min(a,x)) <= threshold
               for x,y,_ in kept):
            kept.append((a,b,score))
    return sorted(kept)
