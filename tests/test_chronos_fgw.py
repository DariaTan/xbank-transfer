import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from unittest import TestCase

import pandas as pd
import yaml

from cross_schema.apply_mapping import FrozenSchemaMapping
from training.chronos_fgw import publish
from training.infer_chronos_mbd_raw import _manifest


def fixture(root):
    configs=root/'configs';(configs/'data').mkdir(parents=True);(configs/'models').mkdir()
    targets=root/'targets.parquet';transactions=root/'transactions.parquet'
    pd.DataFrame({'id':['a','a'],'col_1':['2023-01-01','2024-01-01']}).to_parquet(targets,index=False)
    pd.DataFrame({'id':['a'],'col_1':['2022-01-01'],'col_12':[1.]}).to_parquet(transactions,index=False)
    maps={}
    for e in ('xbank','xbank_fgw_v2'):
        mapping=root/f'{e}.json';mapping.write_text(json.dumps({'mapping':{'col_1':'event_time','col_12':'amount'}}));maps[e]=mapping
        (configs/'data'/f'{e}.yaml').write_text(yaml.safe_dump({'paths':{'targets':str(targets),'transactions':str(transactions)},'schema_mapping':str(mapping)}))
    (configs/'models/downstream.yaml').write_text('inference: {history_window_months: 12}\n')
    source=root/'embeds/xbank/zero_shot/chronos2';source.mkdir(parents=True)
    dates=['2023-01-01','2024-01-01']
    for date in dates:pd.DataFrame({'inn':['a'],'date':[date],'emb_0':[1.]}).to_parquet(source/f'{date}.parquet',index=False)
    manifest=_manifest(transactions,targets,dates,12,32,'col_12',FrozenSchemaMapping(maps['xbank']).sha256)
    (source/'run_manifest.json').write_text(json.dumps(manifest))
    return configs,source,maps,transactions


def test_reuse_is_hardlinked_idempotent_and_preserves_original():
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'XBANK_DATA_ROOT':temp}):
        configs,source,maps,_=fixture(Path(temp));original=(source/'run_manifest.json').read_bytes()
        destination=publish(configs)
        for p in source.glob('*.parquet'):assert os.path.samefile(p,destination/p.name)
        assert (source/'run_manifest.json').read_bytes()==original
        assert json.loads((destination/'embedding_reuse.json').read_text())['effective_input']=={'event_time':'col_1','amount':'col_12'}
        publish(configs)


def test_different_effective_input_requires_inference():
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'XBANK_DATA_ROOT':temp}):
        configs,_,maps,_=fixture(Path(temp));maps['xbank_fgw_v2'].write_text(json.dumps({'mapping':{'col_1':'event_time','col_11':'amount'}}))
        with TestCase().assertRaisesRegex(ValueError,'effective input differs'):publish(configs)


def test_changed_source_and_existing_foreign_results_are_rejected():
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'XBANK_DATA_ROOT':temp}):
        configs,source,_,transactions=fixture(Path(temp))
        destination=Path(temp)/'embeds/xbank_fgw_v2/zero_shot/chronos2';destination.mkdir(parents=True)
        p=destination/'2023-01-01.parquet';p.write_bytes((source/p.name).read_bytes());before=p.read_bytes()
        with TestCase().assertRaisesRegex(ValueError,'overwrite'):publish(configs)
        assert p.read_bytes()==before
        transactions.touch()
        with TestCase().assertRaisesRegex(ValueError,'no longer matches'):publish(configs)


if __name__ == '__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn(); print(f'PASS {name}',flush=True)
