"""Dictionary-free cross-schema alignment (paper Sec. 5): column passports,
semi-relaxed FGW matching, the MBD testbed, and the frozen-mapping artifact.

Current matcher formulas and validation protocol: see FGW.md; dataset rebuilding: see REBUILD.md.
The example output below is historical (passport-v1), not an FGW result.

Targets are out of scope by design: only the transaction tables' FEATURE
columns are ever profiled, matched, or permuted. Target files are never
opened -- labels stay the untouched evaluation axis.


ВСЁ сразу: 3 тестбеда + профили + матчинг xbank→mbd + заморозка
TODO: выполнить это до инфреенса
PYTHONPATH=src python -m cross_schema.column_profiles_vFGW --sample-rows 2000000 --structure-rows 200000 --testbed-variant all --out-dir data/profiles_vFGW


ПООЧЕРЁДНО:
1) тестбед folds: проверка механики

PYTHONPATH=src python -m cross_schema.column_profiles_vFGW --testbed-only --sample-rows 2000000 --structure-rows 200000 --testbed-variant folds --out-dir data/profiles_vFGW
Результат старого запуска passport-v1: recovery: 12/12 exact (hungarian) - матчер вслепую восстановил все соответствия MBD↔MBD.


2) агрегация дневной таблицы

PYTHONPATH=src python -m data.mbd_adapter --freq daily \
  --mbd-root /home/stsix/xbank-transfer/data/mbd \
  --temp-dir /home/stsix/xbank-transfer/data/duckdb_tmp \
  --folds 0 \
  --transactions-out /home/stsix/xbank-transfer/data/mbd_daily/transactions_adapted_daily.parquet \
  --targets-out /home/stsix/xbank-transfer/data/mbd_daily/targets.parquet

  Пояснения по флагам: 
--mbd-root и --temp-dir \чтобы адаптер работал на хосте, а не только в контейнере; 
--folds 0  для тестбеда достаточно одного фолда 
путь выхода совпадает с дефолтом MBD_ADAPTED_DAILY_DEFAULT, поэтому шаг 4 не потребует --daily-path.
Ожидание: две строки wrote ..., несколько минут.


3) тестбед daily_folds: дневные агрегации фолдов MBD 0 и 1

PYTHONPATH=src python -m cross_schema.column_profiles_vFGW --testbed-only --sample-rows 2000000 --structure-rows 200000 --testbed-variant daily_folds --out-dir data/profiles_vFGW
Результат старого запуска passport-v1: recovery: 12/12 exact (hungarian) - матчер вслепую восстановил все соответствия mbd_daily→mbd_daily


4) mbd_daily→mbd - сдвиг зернистости (нужна таблица из адаптера, шаг 2)
PYTHONPATH=src python -m cross_schema.column_profiles_vFGW --testbed-only --sample-rows 2000000 --structure-rows 200000 --testbed-variant daily --out-dir data/profiles_vFGW
Результат старого запуска passport-v1: recovery: 12/12 exact over real fields (hungarian) - матчер вслепую восстановил все соответствия mbd_daily→mbd_daily


5) ТОЛЬКО матчинг xbank→mbd + заморозка (тестбеды уже проверены):
PYTHONPATH=src python -m cross_schema.column_profiles_vFGW --sample-rows 2000000 --structure-rows 200000 --no-testbed --out-dir data/profiles_vFGW



PYTHONPATH=src python -m cross_schema.column_profiles_vFGW \
  --testbed-variant all \
  --sample-rows 2000000 \
  --structure-rows 200000 \
  --out-dir data/profiles_vFGW
[folds_a] materializing source (sample_rows=2000000) ...
[folds_a] profiling columns ...
[folds_a] building dependency matrix on 200,000 rows ...
[folds_a] done in 69.8s
[folds_b] materializing source (sample_rows=2000000) ...
[folds_b] profiling columns ...
[folds_b] building dependency matrix on 200,000 rows ...
[folds_b] done in 70.3s
[folds] recovery: 12/12; solver=CONVERGED
[daily_folds_a] materializing source (sample_rows=2000000) ...
[daily_folds_a] profiling columns ...
[daily_folds_a] building dependency matrix on 200,000 rows ...
[daily_folds_a] done in 108.4s
[daily_folds_b] materializing source (sample_rows=2000000) ...
[daily_folds_b] profiling columns ...
[daily_folds_b] building dependency matrix on 200,000 rows ...
[daily_folds_b] done in 98.5s
[daily_folds] recovery: 12/12; solver=CONVERGED
[daily_a] materializing source (sample_rows=2000000) ...
[daily_a] profiling columns ...
[daily_a] building dependency matrix on 200,000 rows ...
[daily_a] done in 124.6s
[daily_b] materializing source (sample_rows=2000000) ...
[daily_b] profiling columns ...
[daily_b] building dependency matrix on 200,000 rows ...
[daily_b] done in 58.3s
[daily] recovery: 12/12; solver=CONVERGED
[xbank] materializing source (sample_rows=2000000) ...
[xbank] profiling columns ...
[xbank] building dependency matrix on 200,000 rows ...
[xbank] done in 88.7s
[mbd] materializing source (sample_rows=2000000) ...
[mbd] profiling columns ...
[mbd] building dependency matrix on 200,000 rows ...
[mbd] done in 53.1s

=== xbank -> MBD input fields (source reuse allowed) ===
    mbd_field xbank_col   status  distance  fgw_score  second_score  column_margin  candidate_count ambiguity  source_reuse_count
       amount    col_12   mapped  0.105775        1.0           0.0            1.0                2 SEPARATED                   1
   event_type     col_2   mapped  0.041055        1.0           0.0            1.0                4 SEPARATED                   4
event_subtype     col_2   mapped  0.063166        1.0           0.0            1.0                4 SEPARATED                   4
     currency     col_6   mapped  0.034383        1.0           0.0            1.0               12 SEPARATED                   1
   src_type11     col_3   mapped  0.241901        1.0           0.0            1.0                4 SEPARATED                   2
   src_type12     col_3   mapped  0.141646        1.0           0.0            1.0                2 SEPARATED                   2
   dst_type11     col_2   mapped  0.050684        1.0           0.0            1.0                4 SEPARATED                   4
   dst_type12     col_2   mapped  0.338297        1.0           0.0            1.0                2 SEPARATED                   4
   src_type21      None unfilled       NaN        NaN           NaN            NaN                0       NaN                   0
   src_type22    col_10   mapped  0.345933        1.0           0.0            1.0                3 SEPARATED                   2
   src_type31      None unfilled       NaN        NaN           NaN            NaN                0       NaN                   0
   src_type32    col_10   mapped  0.362506        1.0           0.0            1.0                3 SEPARATED                   2
Solver: CONVERGED; wrote data/profiles_vFGW
"""