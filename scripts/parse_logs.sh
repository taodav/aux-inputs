#!/usr/bin/env bash

# This script should be run by piping the output of the below to this script.
# ls -l *.err | awk '{ print $9 "," $5 }'
# This will list all files that end in .err, and print out the filename.err,size
# one file for each row.

IDS_TO_RUN=()
while IFS= read -r line; do
  arrIN=(${line//,/ })
  FSIZE=(${arrIN[1]})
  FNAME=(${arrIN[0]})
  if [[ $1 != $FSIZE ]]
  then
    # get just the file name excluding extension
    arrIN1=(${FNAME//./ })

    # get the ID
    arrIN2=(${arrIN1[0]//-/ })
    runID=(${arrIN2[-1]})
    IDS_TO_RUN+=($runID)
    printf '%s\n' "$line"
  fi
done

printf -v joined '%s,' "${IDS_TO_RUN[@]}"
echo "${joined%,}"