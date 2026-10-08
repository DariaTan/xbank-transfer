import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import yaml

from training.tune_mlp import (ProbeMLP, blocks, fit_standardizer, job, predict,
                               queue_jobs, checked_files)
from training.tune_lightgbm_chronos import metrics, prepare_cache, split_rows


def fixture(root, mbd=True):
    configs = root / 'configs'; (configs / 'data').mkdir(parents=True)
    (configs / 'models').mkdir()
    dates = ([f'2022-{m:02d}-01' for m in range(1, 13)] if mbd else
             [f'2023-{m:02d}-01' for m in range(1, 11)] + ['2024-01-01', '2024-02-01'])
    targets = []
    for date in dates:
        for client in range(100):
            row = {'id': f'c{client}', 'col_1': date,
                   **{f'col_{t}': (client // 5) % 2 for t in range(2, 6)}}
            if mbd: row['fold'] = client % 5
            targets.append(row)
            if not mbd: targets.append({**row, 'col_2': 1-row['col_2']})
    target = root / 'targets.parquet'; pd.DataFrame(targets).to_parquet(target, index=False)
    evaluations = ['mbd_raw'] if mbd else ['xbank', 'xbank_fgw_v2']
    for e in evaluations:
        filename = 'mbd.yaml' if e == 'mbd_raw' else f'{e}.yaml'
        (configs / 'data' / filename).write_text(yaml.safe_dump({'name':e, 'evaluation_name':e, 'paths':{'targets':str(target)}}))
        directory = root / 'embeds' / e / 'zero_shot/chronos2'; directory.mkdir(parents=True)
        for date in dates:
            # Reverse the FGW order and omit two clients: paired alignment must not use row offsets as keys.
            clients = list(range(100)) if e != 'xbank_fgw_v2' else list(range(2, 100))[::-1]
            pd.DataFrame([{'inn':f'c{i}', 'date':date, 'emb_0':float((i//5)%2),
                           'emb_1':float(i%5), 'emb_2':1.} for i in clients]).to_parquet(directory/f'{date}.parquet',index=False)
    probe = {'target_cols':['col_3','col_4','col_5'], 'train':{'start':'2023-01','end':'2023-12'},
             'test':{'start':'2024-01','end':'2024-02'}, 'val_frac':.4, 'seed':0}
    (configs / 'models/downstream.yaml').write_text(yaml.safe_dump({'probe':probe}))
    cfg = {'seed':42,'batch_size':64,'max_epochs':2,'patience':1,'tune_client_cap':100,
           'threads':1,'clip_standardized_features':10., 'candidates':[
               {'hidden':8,'lr':.01,'dropout':0.,'weight_decay':.0001,'class_weight':'none'}]}
    mlp = configs / 'models/downstream_mlp.yaml'; mlp.write_text(yaml.safe_dump(cfg))
    return configs, mlp, target


def device():
    selected = os.environ.get('MLP_SMOKE_DEVICE','cpu')
    if selected == 'gpu': assert torch.cuda.is_available(), 'GPU smoke cannot fall back to CPU'
    return selected


def test_standardizer_does_not_see_validation_or_test():
    x = np.array([[1,7],[3,7],[1e9,-1e9]],np.float32)
    center,scale = fit_standardizer(x,np.arange(3),np.array([0,1]),1)
    np.testing.assert_allclose(center,[2,7]); np.testing.assert_allclose(scale,[1,1e-6])


def test_batches_visit_each_row_once_and_are_seeded():
    rows=np.arange(137)
    a=np.concatenate(list(blocks(rows,16,np.random.default_rng(42))))
    b=np.concatenate(list(blocks(rows,16,np.random.default_rng(42))))
    np.testing.assert_array_equal(np.sort(a),rows);np.testing.assert_array_equal(a,b)


def test_queue_is_unique_and_complete():
    jobs=queue_jobs();assert len(jobs)==28;assert len(set(jobs))==28
    assert ('xbank_pair','mbd','chronos2') in jobs
    assert not any(e=='mbd_raw' and s=='mbd_daily' for e,s,m in jobs)
    assert jobs.index(('xbank_pair','mbd','chronos2'))%2==1
    assert jobs.index(('mbd_raw','mbd','chronos2'))%2!=jobs.index(('mbd_daily','mbd','chronos2'))%2
    assert len(jobs[::2])==len(jobs[1::2])==14


def test_mbd_five_folds_resume_saved_model_and_preserve_lightgbm():
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'XBANK_DATA_ROOT':temp}):
        root=Path(temp);configs,mlp,target=fixture(root)
        parent=root/'downstream/mbd_raw/zero_shot/chronos2';parent.mkdir(parents=True)
        sentinel=parent/'old_lightgbm.txt';sentinel.write_text('do not change')
        job('mbd_raw','mbd','chronos2',configs,mlp,device())
        output=parent/'mlp_hpo';results=pd.read_csv(output/'results_all_folds.csv')
        assert len(results)==20 and set(results.target)=={'col_2','col_3','col_4','col_5'}
        assert (pd.read_csv(output/'results_aggregated.csv').n_evaluations==5).all()
        assert not (output/'_feature_cache').exists();assert sentinel.read_text()=='do not change'
        checkpoints={p:p.stat().st_mtime_ns for p in output.glob('fold*/*.pt')}
        job('mbd_raw','mbd','chronos2',configs,mlp,device())
        assert checkpoints=={p:p.stat().st_mtime_ns for p in checkpoints}
        # Restore a checkpoint and independently recompute both reported test metrics.
        files,_=checked_files('mbd_raw','mbd','chronos2',configs)
        cache=root/'verify_cache';saved=prepare_cache(cache,target,files,[f'col_{i}' for i in range(2,6)],True)
        values=np.memmap(cache/'features.f32',dtype=np.float32,mode='r',shape=(saved['capacity'],saved['n_features']))
        rows=pd.read_parquet(cache/'rows.parquet');_,_,test,_,_=split_rows(rows,True,0,{},42,100)
        saved_model=torch.load(output/'fold0/model_col_2.pt',map_location='cpu',weights_only=True)
        net=ProbeMLP(saved_model['n_features'],saved_model['params']['hidden'],saved_model['params']['dropout'])
        net.load_state_dict(saved_model['state_dict'])
        score=predict(net,values,np.arange(len(rows)),test,(saved_model['center'].numpy(),saved_model['scale'].numpy()),torch.device('cpu'),64,10.)
        recomputed=metrics(rows.col_2.to_numpy()[test],score)
        recorded=json.loads((output/'fold0/metrics_col_2.json').read_text())['test_metrics']
        for key in ('pr_auc','roc_auc'): assert abs(recomputed[key]-recorded[key])<1e-5
        changed=yaml.safe_load(mlp.read_text());changed['seed']=43;mlp.write_text(yaml.safe_dump(changed))
        with TestCase().assertRaisesRegex(ValueError,'settings differ'):job('mbd_raw','mbd','chronos2',configs,mlp,device())


def test_xbank_paired_alignment_excludes_col2_and_no_cpu_fallback():
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'XBANK_DATA_ROOT':temp}):
        root=Path(temp);configs,mlp,_=fixture(root,False)
        job('xbank_pair','mbd','chronos2',configs,mlp,device())
        comparison=pd.read_csv(root/'downstream/xbank/zero_shot/chronos2/mlp_hpo/mapping_comparison.csv')
        assert set(comparison.target)=={'col_3','col_4','col_5'}
        assert (comparison.n_rows_original==196).all()
        assert (comparison.n_rows_original==comparison.n_rows_fgw_v2).all()
        assert (comparison.delta_pr_auc==0).all() and (comparison.delta_roc_auc==0).all()
        for t in ('col_3','col_4','col_5'):
            checkpoints=[torch.load(root/f'downstream/{e}/zero_shot/chronos2/mlp_hpo/calendar/model_{t}.pt',weights_only=True,map_location='cpu')
                         for e in ('xbank','xbank_fgw_v2')]
            for k in checkpoints[0]['state_dict']:
                torch.testing.assert_close(checkpoints[0]['state_dict'][k],checkpoints[1]['state_dict'][k],rtol=0,atol=0)
        with patch('torch.cuda.is_available',return_value=False),TestCase().assertRaisesRegex(RuntimeError,'CPU fallback'):
            job('xbank_pair','mbd','chronos2',configs,mlp,'gpu')


if __name__ == '__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn(); print(f'PASS {name}',flush=True)
