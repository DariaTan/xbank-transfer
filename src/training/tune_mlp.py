"""Memory-bounded frozen-embedding MLP with leakage-safe, resumable probes.

Uses the same MBD outer folds and Xbank paired calendar/client splits as the
LightGBM probes. HPO and epoch selection use validation average precision;
the selected network is reinitialized and refit on train+validation before
one test evaluation. Scaling is fitted on train for selection, then on dev
for final refit. No test statistics, negative sampling or cross-target labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from training.paths import downstream_dir, embedding_dir, load_data_config
from training.train_downstream import load_config
from training.tune_lightgbm_chronos import (KEYS, FORMAT_VERSION, _release_pages, atomic_json,
    check_manifest, metrics, prepare_cache, signature, split_rows)
from training.tune_lightgbm_xbank_paired import paired_cohort

MODELS = ('coles', 'cotic', 'thp', 'nep', 'mlm', 'chronos2')
TARGETS_XBANK = ('col_3', 'col_4', 'col_5')


def data_config_path(config_dir, evaluation):
    # Historical production filename differs from its evaluation namespace.
    return config_dir / 'data' / ('mbd.yaml' if evaluation == 'mbd_raw' else f'{evaluation}.yaml')


def amp_dtype(device):
    return torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


class ProbeMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def disk_order(offsets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return rows[np.argsort(offsets[rows], kind='stable')]


def blocks(rows: np.ndarray, size: int, rng=None):
    starts = np.arange(0, len(rows), size)
    if rng is not None: rng.shuffle(starts)
    for start in starts:
        part = rows[start:start + size]
        # Shuffle within disk-local blocks rather than random-reading a
        # 34-GB panel on every SGD step. Every row appears once per epoch.
        yield rng.permutation(part) if rng is not None else part


def fit_standardizer(values, offsets, rows, batch_size=16384):
    total = np.zeros(values.shape[1], np.float64)
    squares = np.zeros_like(total)
    n = 0
    for ids in blocks(disk_order(offsets, rows), batch_size):
        x = np.array(values[offsets[ids]], dtype=np.float64, copy=True)
        total += x.sum(axis=0); squares += np.square(x).sum(axis=0); n += len(x)
        if isinstance(values, np.memmap): _release_pages(values)
    if not n: raise ValueError('cannot fit normalization on empty training set')
    center = total / n
    scale = np.sqrt(np.maximum(squares / n - center**2, 1e-12))
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError('non-finite training normalization')
    return center.astype(np.float32), scale.astype(np.float32)


def normalized_batch(values, offsets, ids, center, scale, device, clip):
    # Read physical offsets in order, then restore the logical minibatch order.
    order = np.argsort(offsets[ids], kind='stable')
    x = np.array(values[offsets[ids[order]]], dtype=np.float32, copy=True)
    x = x[np.argsort(order)]
    if isinstance(values, np.memmap): _release_pages(values)
    x = torch.from_numpy(x).to(device)
    return ((x - center) / scale).clamp(-clip, clip)


@torch.inference_mode()
def predict(model, values, offsets, rows, norm, device, batch_size, clip):
    model.eval()
    center, scale = (torch.as_tensor(a, device=device) for a in norm)
    order = np.argsort(offsets[rows], kind='stable')
    scores = np.empty(len(rows), np.float32)
    for positions in blocks(order, batch_size):
        x = normalized_batch(values, offsets, rows[positions], center, scale, device, clip)
        # Float32 inference avoids bf16 logit quantization/ties in ranking metrics.
        logits = model(x)
        scores[positions] = logits.float().cpu().numpy()
    if not np.isfinite(scores).all(): raise ValueError('non-finite MLP predictions')
    return scores


def fit(values, offsets, train, labels, norm, params, cfg, device, seed, *, val=None, epochs=None, logical_order=False):
    seed_all(seed)
    model = ProbeMLP(values.shape[1], params['hidden'], params['dropout']).to(device)
    prevalence = float(labels[train].mean())
    if not 0 < prevalence < 1: raise ValueError('training split lacks one binary class')
    balanced = params['class_weight'] == 'balanced'
    if params['class_weight'] not in ('none', 'balanced'): raise ValueError('unknown class weighting')
    with torch.no_grad():
        model.net[-1].bias.fill_(0. if balanced else np.log(prevalence / (1 - prevalence)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - prevalence) / prevalence, device=device) if balanced else None)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp_dtype(device) == torch.float16)
    center, scale = (torch.as_tensor(a, device=device) for a in norm)
    # Paired mappings must have identical SGD batches even when their parquet
    # row order differs. Logical row indices refer to the same paired cohort.
    ordered = np.sort(train) if logical_order else disk_order(offsets, train)
    best_score, best_epoch, bad = -1., 0, 0
    for epoch in range(epochs or cfg['max_epochs']):
        model.train(); total, count = 0., 0
        rng = np.random.default_rng(seed + epoch)
        for ids in blocks(ordered, cfg['batch_size'], rng):
            x = normalized_batch(values, offsets, ids, center, scale, device, cfg['clip_standardized_features'])
            y = torch.from_numpy(np.asarray(labels[ids], np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == 'cuda'):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss): raise ValueError('non-finite MLP loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(ids); count += len(ids)
        if val is not None:
            scores = predict(model, values, offsets, val, norm, device, cfg['batch_size'], cfg['clip_standardized_features'])
            score = metrics(labels[val], scores)['pr_auc']
            if score > best_score:
                best_score, best_epoch, bad = score, epoch + 1, 0
            else: bad += 1
            print(f'epoch={epoch+1} train_loss={total/count:.6f} val_PR-AUC={score:.6f} best_epoch={best_epoch}', flush=True)
            if bad >= cfg['patience']: break
        else:
            print(f'refit epoch={epoch+1}/{epochs} train_loss={total/count:.6f}', flush=True)
    return model, {'validation_pr_auc': best_score, 'best_epochs': best_epoch}


def checked_files(evaluation: str, source: str, model: str, config_dir: Path):
    directory = embedding_dir('/app/data/embeds', evaluation, source, model)
    files = sorted(directory.glob('*.parquet'))
    if len(files) != 12: raise ValueError(f'{directory}: expected 12 published dates, found {len(files)}')
    manifest = directory / 'run_manifest.json'
    provenance = json.loads(manifest.read_text()) if manifest.exists() else None
    # Historical raw-source runs do not all have inference manifests; their
    # immutable file signatures remain explicit and the cache validates keys.
    cfg = load_data_config(data_config_path(config_dir, evaluation))
    dates = set(pd.read_parquet(cfg['paths']['targets'], columns=['col_1']).col_1.astype(str))
    if {p.stem for p in files} != dates: raise ValueError('target dates and embeddings differ')
    if model != 'chronos2' and (provenance is not None or source == 'mbd_daily'):
        if not provenance or provenance.get('checkpoint_source') != source or provenance.get('evaluation_name') != evaluation:
            raise ValueError('inference checkpoint/evaluation provenance mismatch')
    return files, {'files': [signature(p) for p in files], 'inference_manifest': provenance}


def run_one(evaluation, source, model, fold, files, provenance, cfg, config_dir, device, paired=None, paired_sources=None):
    is_mbd = evaluation.startswith('mbd_')
    targets = [f'col_{i}' for i in range(2, 6)] if is_mbd else list(TARGETS_XBANK)
    data_cfg = load_data_config(data_config_path(config_dir, evaluation))
    root = downstream_dir('/app/data/downstream', evaluation, source, model) / 'mlp_hpo'
    output = root / f'fold{fold}' if is_mbd else root / 'calendar'
    output.mkdir(parents=True, exist_ok=True)
    probe = yaml.safe_load((config_dir / 'models/downstream.yaml').read_text())['probe']
    manifest = {'format_version': 1, 'architecture': 'mlp', 'model': model, 'checkpoint_source': source,
                'implementation_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in ('tune_mlp.py', 'tune_lightgbm_chronos.py', 'tune_lightgbm_xbank_paired.py', 'target_io.py')},
                'evaluation_name': evaluation, 'test_fold': fold if is_mbd else None, 'targets': targets,
                'input_provenance': provenance, 'target_file': signature(Path(data_cfg['paths']['targets'])),
                'hpo': cfg, 'device_type': device.type, 'negative_sampling': 'none',
                'selection_metric': 'validation average precision', 'normalization': 'train-only selection; dev-only refit',
                'reported_metrics': ['pr_auc', 'roc_auc'], 'calendar_probe': None if is_mbd else probe,
                'batch_order': 'paired logical cohort' if paired is not None else 'disk-local',
                'paired_sources': paired_sources}
    if paired is not None:
        manifest['paired_cohort_sha256'] = hashlib.sha256(pd.util.hash_pandas_object(paired[KEYS + targets], index=False).values.tobytes()).hexdigest()
    check_manifest(output / 'run_manifest.json', manifest)
    pending = [t for t in targets if not ((output / f'metrics_{t}.json').exists() and (output / f'model_{t}.pt').exists())]
    if not pending:
        print(f'COMPLETE skipping {evaluation}/{source}/{model}/fold{fold}', flush=True); return
    cache = root / '_feature_cache'
    saved = prepare_cache(cache, Path(data_cfg['paths']['targets']), files, targets, is_mbd)
    rows = pd.read_parquet(cache / 'rows.parquet')
    offsets = np.arange(len(rows), dtype=np.int64)
    if paired is not None:
        offsets = pd.MultiIndex.from_frame(rows[KEYS]).get_indexer(pd.MultiIndex.from_frame(paired[KEYS]))
        if (offsets < 0).any() or paired.duplicated(KEYS).any(): raise ValueError('invalid paired feature alignment')
        rows = rows.iloc[offsets].reset_index(drop=True)
        if not np.array_equal(rows[targets], paired[targets]): raise ValueError('paired labels differ')
    train, val, test, tune, dev = split_rows(rows, is_mbd, fold, probe, cfg['seed'], cfg['tune_client_cap'])
    if not is_mbd:
        clients = np.sort(rows.iloc[train].id.unique())
        selected = np.random.default_rng(cfg['seed']).choice(clients, min(cfg['tune_client_cap'], len(clients)), replace=False)
        tune = train[rows.iloc[train].id.isin(selected).to_numpy()]
    labels = {t: rows[t].to_numpy(dtype=np.int8) for t in targets}
    for t in targets:
        if any(set(np.unique(labels[t][ids])) != {0, 1} for ids in (tune, train, val, test)):
            raise ValueError(f'{t}: split lacks one binary class')
    atomic_json(output / 'cohort.json', {'train_rows':len(train), 'val_rows':len(val), 'test_rows':len(test),
                'tune_rows':len(tune), 'n_features':saved['n_features'], 'cohort_sha256': saved['cohort_sha256']})
    print(f'{evaluation}/{source}/{model}/fold{fold}: train={len(train)} tune={len(tune)} val={len(val)} test={len(test)} features={saved["n_features"]} device={device}', flush=True)
    del rows; gc.collect()
    values = np.memmap(cache / 'features.f32', dtype=np.float32, mode='r', shape=(saved['capacity'], saved['n_features']))
    def normalization(name, ids):
        path = output / f'normalization_{name}.npz'
        if path.exists():
            with np.load(path) as n: return n['center'], n['scale']
        norm = fit_standardizer(values, offsets, ids, cfg['batch_size'])
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, center=norm[0], scale=norm[1])
        os.replace(temporary, path)
        return norm
    selections = {}
    needed = [t for t in pending if not (output / f'selection_{t}.json').exists()]
    if needed:
        norm = normalization('train', train)
        for t in needed:
            trials, best = [], None
            for i, params in enumerate(cfg['candidates']):
                model_net, choice = fit(values, offsets, tune, labels[t], norm, params, cfg, device, cfg['seed'] + i, val=val, logical_order=paired is not None)
                trial = {'trial':i, 'val_pr_auc':choice['validation_pr_auc'], 'best_epochs':choice['best_epochs'], **params}
                trials.append(trial)
                if best is None or trial['val_pr_auc'] > best['val_pr_auc']: best = trial
                del model_net; gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
            # Re-select training duration with selected hyperparameters on
            # the full outer training set, just as the LightGBM probe does.
            params = cfg['candidates'][best['trial']]
            model_net, choice = fit(values, offsets, train, labels[t], norm, params, cfg, device, cfg['seed'], val=val, logical_order=paired is not None)
            del model_net
            selection = {'best_params':params, 'best_epochs':choice['best_epochs'],
                         'full_val_pr_auc':choice['validation_pr_auc'], 'tune_val_pr_auc':best['val_pr_auc']}
            if choice['best_epochs'] < 1: raise ValueError('no valid selected epoch')
            pd.DataFrame(trials).to_csv(output / f'trials_{t}.csv', index=False)
            atomic_json(output / f'selection_{t}.json', selection)
            selections[t] = selection
    for t in pending:
        selections[t] = json.loads((output / f'selection_{t}.json').read_text())
    norm = normalization('dev', dev)
    for t in pending:
        choice = selections[t]
        model_net, _ = fit(values, offsets, dev, labels[t], norm, choice['best_params'], cfg, device, cfg['seed'], epochs=choice['best_epochs'], logical_order=paired is not None)
        # No early stopping or HPO is allowed after this test access.
        scores = predict(model_net, values, offsets, test, norm, device, cfg['batch_size'], cfg['clip_standardized_features'])
        result = metrics(labels[t][test], scores)
        temporary = output / f'model_{t}.pt.tmp'
        torch.save({'state_dict':{k:v.cpu() for k,v in model_net.state_dict().items()},
                    'center':torch.from_numpy(norm[0]), 'scale':torch.from_numpy(norm[1]),
                    'clip':cfg['clip_standardized_features'], 'n_features':saved['n_features'],
                    'params':choice['best_params'], 'seed':cfg['seed']}, temporary)
        os.replace(temporary, output / f'model_{t}.pt')
        atomic_json(output / f'metrics_{t}.json', {'architecture':'mlp', 'target':t, **choice,
                    'training_device':device.type, 'test_metrics':result,
                    'test_access':'once after hyperparameter/epoch selection and train+val refit'})
        print(f'TEST {t}: PR-AUC={result["pr_auc"]:.6f} ROC-AUC={result["roc_auc"]:.6f}', flush=True)
        del model_net, scores; gc.collect()
    del values; gc.collect()


def summarize(evaluation, source, model, cleanup=True):
    root = downstream_dir('/app/data/downstream', evaluation, source, model) / 'mlp_hpo'
    is_mbd = evaluation.startswith('mbd_')
    records, reference = [], None
    for fold in range(5) if is_mbd else [None]:
        directory = root / f'fold{fold}' if is_mbd else root / 'calendar'
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        settings = {k:v for k,v in manifest.items() if k != 'test_fold'}
        if reference is not None and settings != reference: raise ValueError('MLP fold settings differ')
        reference = settings
        for t in manifest['targets']:
            if not (directory / f'model_{t}.pt').exists(): raise ValueError('missing saved MLP')
            result = json.loads((directory / f'metrics_{t}.json').read_text())
            records.append({'architecture':'mlp', 'evaluation_name':evaluation, 'checkpoint_source':source,
                            'model':model, 'target':t, 'test_fold':fold, **result['test_metrics']})
    rows = pd.DataFrame(records)
    aggregate = rows.groupby('target', sort=True).agg(n_evaluations=('pr_auc','size'),
        pr_auc_mean=('pr_auc','mean'),pr_auc_std=('pr_auc','std'),roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std')).reset_index()
    macro = pd.DataFrame([{'mean_pr_auc_across_targets':aggregate.pr_auc_mean.mean(),
                           'mean_roc_auc_across_targets':aggregate.roc_auc_mean.mean()}])
    for name, frame in [('results_all_folds.csv',rows),('results_aggregated.csv',aggregate),('model_macro.csv',macro)]:
        temporary = root / f'{name}.tmp'; frame.to_csv(temporary,index=False); os.replace(temporary,root/name)
    if cleanup and (root/'_feature_cache').is_dir():
        saved=json.loads((root/'_feature_cache/manifest.json').read_text())
        if saved['source']['format_version'] != FORMAT_VERSION: raise ValueError('unknown temporary cache')
        shutil.rmtree(root/'_feature_cache')
    print(aggregate.to_string(index=False), flush=True)


def job(evaluation, source, model, config_dir, mlp_config, device):
    cfg=yaml.safe_load(Path(mlp_config).read_text())
    if min(cfg['batch_size'],cfg['max_epochs'],cfg['patience'],cfg['tune_client_cap'],cfg['threads'])<1 or not cfg['candidates']:
        raise ValueError('invalid MLP resource/search settings')
    if device == 'gpu' and not torch.cuda.is_available(): raise RuntimeError('GPU required; CPU fallback is forbidden')
    dev=torch.device('cuda:0' if device=='gpu' else 'cpu');torch.set_num_threads(cfg['threads'])
    if dev.type=='cuda': print(f'GPU: {torch.cuda.get_device_name(0)}',flush=True)
    evaluations=('xbank','xbank_fgw_v2') if evaluation=='xbank_pair' else (evaluation,)
    files,provenance={},{}
    for e in evaluations: files[e],provenance[e]=checked_files(e,source,model,config_dir)
    paired=paired_sources=None
    if evaluation=='xbank_pair':
        paths={load_data_config(data_config_path(config_dir,e))['paths']['targets'] for e in evaluations}
        if len(paths)!=1: raise ValueError('paired mappings use different targets')
        paired,_,_,_=paired_cohort(model,Path(paths.pop()),load_config(str(config_dir/'models/downstream.yaml')),files)
        paired=paired[KEYS+list(TARGETS_XBANK)].copy();paired['id']=paired.id.astype(str)
        paired_sources=provenance
    for e in evaluations:
        for fold in range(5) if e.startswith('mbd_') else [0]:
            run_one(e,source,model,fold,files[e],provenance[e],cfg,config_dir,dev,paired,paired_sources)
        summarize(e,source,model)
    if evaluation=='xbank_pair':
        roots={e:downstream_dir('/app/data/downstream',e,source,model)/'mlp_hpo' for e in evaluations}
        a,b=[pd.read_csv(roots[e]/'results_all_folds.csv') for e in evaluations]
        paired_result=a.merge(b,on=['target'],suffixes=('_original','_fgw_v2'),validate='one_to_one')
        for metric in ('n_rows','n_positive','prevalence'):
            if not paired_result[f'{metric}_original'].equals(paired_result[f'{metric}_fgw_v2']): raise ValueError('paired MLP test cohorts differ')
        for metric in ('pr_auc','roc_auc'): paired_result[f'delta_{metric}']=paired_result[f'{metric}_fgw_v2']-paired_result[f'{metric}_original']
        paired_result.to_csv(roots['xbank']/'mapping_comparison.csv',index=False)


def queue_jobs():
    # Xbank first gives timely institutional-shift results. Keep all folds
    # of a representation on one worker so its cache has a single owner.
    jobs=[('xbank_pair',s,m) for s in ('mbd','mbd_daily') for m in MODELS if m!='chronos2']
    jobs += [('xbank_pair','mbd','chronos2')]
    jobs += [(e,s,m) for e,s in [('mbd_raw','mbd'),('mbd_daily','mbd'),('mbd_daily','mbd_daily')]
             for m in MODELS if not (s=='mbd_daily' and m=='chronos2')]
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',choices=['mbd_raw','mbd_daily','xbank_pair'])
    p.add_argument('--checkpoint-source',choices=['mbd','mbd_daily'],default='mbd')
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--config-dir',type=Path,default=Path('/app/configs'))
    p.add_argument('--mlp-config',default='/app/configs/models/downstream_mlp.yaml')
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--list-jobs',action='store_true');p.add_argument('--worker-index',type=int,choices=[0,1],default=0)
    args=p.parse_args()
    if args.list_jobs:
        for i,entry in enumerate(queue_jobs()):
            if i%2==args.worker_index: print(' '.join(entry))
        return
    if not args.evaluation or not args.model: p.error('--evaluation and --model required')
    if args.evaluation=='mbd_raw' and args.checkpoint_source=='mbd_daily': p.error('daily-source raw embeddings do not exist')
    job(args.evaluation,args.checkpoint_source,args.model,args.config_dir,args.mlp_config,args.device)


if __name__=='__main__': main()
