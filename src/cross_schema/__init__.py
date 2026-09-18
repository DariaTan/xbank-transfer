"""Dictionary-free cross-schema alignment (paper Sec. 5): column passports,
similarity matching, the MBD testbed, and the frozen-mapping artifact.

Targets are out of scope by design: only the transaction tables' FEATURE
columns are ever profiled, matched, or permuted. Target files are never
opened -- labels stay the untouched evaluation axis.


ВСЁ сразу: 3 тестбеда + профили + матчинг xbank→mbd + якоря + заморозка
TODO: выполнить это до инфреенса
PYTHONPATH=src python -m cross_schema.column_profiles --sample-rows 2000000 --testbed-variant all


ПООЧЕРЁДНО:
1) тестбед folds: проверка механики

PYTHONPATH=src python -m cross_schema.column_profiles --testbed-only --sample-rows 2000000 --testbed-variant folds
Результат: recovery: 12/12 exact (hungarian) - матчер вслепую восстановил все соответствия MBD↔MBD.


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
путь выхода совпадает с дефолтом MBD_ADAPTED_DAILY_DEFAULT, поэтому шаг 3 не потребует --daily-path.
Ожидание: две строки wrote ..., несколько минут.


3) тестбед daily на свежей таблице mbd_daily→mbd_daily 

PYTHONPATH=src python -m cross_schema.column_profiles --testbed-only --sample-rows 2000000 --testbed-variant daily_folds
Результат: recovery: 12/12 exact (hungarian) - матчер вслепую восстановил все соответствия mbd_daily→mbd_daily 


4) mbd_daily→mbd - сдвиг зернистости (нужна таблица из адаптера, шаг 2)
PYTHONPATH=src python -m cross_schema.column_profiles --testbed-only --sample-rows 2000000 --testbed-variant daily
Результат: recovery: 12/12 exact over real fields (hungarian) - матчер вслепую восстановил все соответствия mbd_daily→mbd_daily 


5) ТОЛЬКО матчинг xbank→mbd + якоря + заморозка (тестбеды уже проверены):
PYTHONPATH=src python -m cross_schema.column_profiles --sample-rows 2000000 --no-testbed 




Вызод такой:
PYTHONPATH=src python -m cross_schema.column_profiles --sample-rows 2000000 --testbed-variant all
Profiling xbank feature columns ...
Profiling MBD feature columns (folds=[0]) ...
Testbed 'folds': fold 0 anonymized vs fold 1 named ...
  recovery: 12/12 exact (hungarian)
Testbed 'daily_folds': daily fold 0 anonymized vs daily fold 1 named ...
  recovery: 12/12 exact (hungarian)
Testbed 'daily': adapted daily table vs raw hourly MBD ...
  recovery: 12/12 exact over real fields (hungarian)
Matching xbank -> MBD ...

=== full correspondence table (xbank -> MBD) ===
xbank_col     kind     mbd_field similarity   status
    col_1     time    event_time               fixed
    col_2 category    dst_type11     0.9546   mapped
    col_3 category    src_type12     0.9106   mapped
    col_4 category                           dropped
    col_5 category      currency     0.9237   mapped
    col_6 category    src_type11     0.7842   mapped
    col_7 category                           dropped
    col_8 category                           dropped
    col_9 category                           dropped
   col_10 category    event_type     0.9403   mapped
   col_13 category                           dropped
   col_14 category                           dropped
   col_15 category                           dropped
   col_16 category                           dropped
   col_11  numeric                           dropped
   col_12  numeric        amount     0.7903   mapped
                   event_subtype            unfilled
                      dst_type12            unfilled
                      src_type21            unfilled
                      src_type22            unfilled
                      src_type31            unfilled
                      src_type32            unfilled

=== anchors ===
xbank_col   expected                                           why_known auto_assigned   verdict
    col_1 event_time             dtype DATE, range 2022-2024 (schema.py)    event_time        OK
    col_2 event_type        report.html: MCC-like, 52 values (schema.py)    dst_type11 AMBIGUOUS
   col_11     amount  one of the only two continuous columns (schema.py)          None      INFO
   col_12     amount the other continuous column; which-is-which unknown        amount      INFO

wrote 7 artifacts to /home/stsix/xbank-transfer/data/profiles; frozen_mapping.json is what inference reads
"""