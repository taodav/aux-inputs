#!/usr/bin/env bash

CURRENT_DATE=$(date "+%d%m%y")
FILE_NAME="snapshot_$CURRENT_DATE.tar.gz"

cd ../
if [ -d "snapshot" ]; then
  rm -r snapshot
fi
mkdir snapshot
mkdir snapshot/results/
mkdir snapshot/analysis/
cp -vr results/ snapshot/results/
cp -vr analysis/ snapshot/analysis/
tar -czvf $FILE_NAME snapshot/

rm -r snapshot