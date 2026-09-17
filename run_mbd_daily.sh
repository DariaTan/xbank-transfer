#!/usr/bin/env bash
cd /app
python -m data.mbd_adapter --freq daily > /app/data/logs/mbd_adapter_daily.log 2>&1
