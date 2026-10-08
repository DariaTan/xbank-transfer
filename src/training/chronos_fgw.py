"""Publish a provenance-checked FGW view when Chronos' effective input is unchanged.

Chronos consumes only event_time and amount. The two current schema mappings
both select col_1/col_12, so copying or recomputing the 16-GB embedding panel
would add no new representation. Hardlinks preserve byte-identical inputs;
the FGW view gets its own mapping manifest and a separate reuse explanation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from cross_schema.apply_mapping import FrozenSchemaMapping
from training.infer_chronos_mbd_raw import _dates, _manifest
from training.paths import embedding_dir, load_data_config, resolve_data_path
from training.tune_lightgbm_chronos import check_manifest, run, signature, summarize


def publish(config_dir: Path) -> Path:
    configs = {v: load_data_config(config_dir / 'data' / f'{v}.yaml')
               for v in ('xbank', 'xbank_fgw_v2')}
    mappings = {v: FrozenSchemaMapping(resolve_data_path(c['schema_mapping']))
                for v, c in configs.items()}
    projection = lambda m: {k: m.field_to_source.get(k) for k in ('event_time', 'amount')}
    if projection(mappings['xbank']) != projection(mappings['xbank_fgw_v2']):
        raise ValueError('Chronos effective input differs; full FGW inference is required')
    if configs['xbank']['paths'] != configs['xbank_fgw_v2']['paths']:
        raise ValueError('Chronos input files differ between mappings')
    source = embedding_dir('/app/data/embeds', 'xbank', 'mbd', 'chronos2')
    destination = embedding_dir('/app/data/embeds', 'xbank_fgw_v2', 'mbd', 'chronos2')
    saved = json.loads((source / 'run_manifest.json').read_text())
    cfg = configs['xbank']
    inf = yaml.safe_load((config_dir / 'models/downstream.yaml').read_text())['inference']
    expected = _manifest(Path(cfg['paths']['transactions']), Path(cfg['paths']['targets']),
                         _dates(Path(cfg['paths']['targets'])), inf['history_window_months'],
                         saved['n_shards'], mappings['xbank'].field_to_source['amount'], mappings['xbank'].sha256)
    if saved != expected:
        raise ValueError('Original Chronos manifest no longer matches current inputs')
    files = sorted(source.glob('*.parquet'))
    if {p.stem for p in files} != set(saved['target_dates']):
        raise ValueError('Original Chronos panel is incomplete')
    for p in files:
        parquet = pq.ParquetFile(p)
        features = [c for c in parquet.schema_arrow.names if c.startswith('emb_')]
        if parquet.metadata.num_rows < 1 or features != [f'emb_{i}' for i in range(len(features))] or not features:
            raise ValueError(f'Invalid published embeddings: {p}')
    mapped = {**expected, 'schema_mapping_sha256': mappings['xbank_fgw_v2'].sha256}
    reuse = {'kind': 'byte_identical_hardlinks', 'reason': 'Both mappings have identical Chronos event_time/amount projection',
             'effective_input': projection(mappings['xbank']), 'source_manifest': signature(source / 'run_manifest.json'),
             'source_files': [signature(p) for p in files]}
    destination.mkdir(parents=True, exist_ok=True)
    check_manifest(destination / 'run_manifest.json', mapped)
    check_manifest(destination / 'embedding_reuse.json', reuse)
    for p in files:
        target = destination / p.name
        if target.exists():
            if not os.path.samefile(p, target):
                raise ValueError(f'Refusing to overwrite a non-identical FGW embedding: {target}')
        else:
            os.link(p, target)
    print(f'Chronos FGW: {len(files)} byte-identical dates published, no feature data duplicated', flush=True)
    return destination


def lightgbm(config_dir: Path, device: str = 'gpu') -> None:
    publish(config_dir)
    original = resolve_data_path('/app/data/downstream/xbank/zero_shot/chronos2/lightgbm_hpo_calendar/run_manifest.json')
    settings = json.loads(original.read_text())
    # Retain the exact baseline budget, seeds and date/client split; do not
    # mutate its results. The new namespace is a separately fitted probe.
    run(str(config_dir / 'data/xbank_fgw_v2.yaml'), str(config_dir / 'models/downstream.yaml'),
        0, device, settings['seed'], settings['trials'], settings['threads'],
        settings['tune_client_cap'], settings['max_rounds'], settings['max_bin'])
    summarize('xbank_fgw_v2', cleanup_cache=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['publish', 'lightgbm'])
    parser.add_argument('--config-dir', type=Path, default=Path('/app/configs'))
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    if args.phase == 'publish':
        publish(args.config_dir)
    else:
        lightgbm(args.config_dir, args.device)


if __name__ == '__main__':
    main()
