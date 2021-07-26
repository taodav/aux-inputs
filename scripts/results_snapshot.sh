#!/usr/bin/env bash

CURRENT_DATE=$(date "+%d%m%y")
FILE_NAME="snapshot_$CURRENT_DATE.tar.gz"

cd ../
if [ -d "snapshot" ]; then
  rm -r snapshot
fi
mkdir snapshot
cp -vr results/ snapshot/
cp -vr analysis/ snapshot/
tar -czvf $FILE_NAME snapshot/

rm -r snapshot