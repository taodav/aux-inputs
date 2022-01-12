#!/usr/bin/env bash

# this script checks each individual log from stdin and checks the last line of the file
# to see if it ends with the word "Saving" (we check if it saved results or not)

# It takes as input ls -1 *.out

IDS_TO_RUN=()
while IFS= read -r line; do
  # get the last line of the file
  LAST_LINE=$( tail -n 1 $line )

  # Split the line by it's spaces
  SPLIT_LINE=(${LAST_LINE// / })

  # If the first word is Saving
  if [[ ${SPLIT_LINE[0]} != "Saving" ]]
  then
    # get the log file name with its extension
    filenameEXT=$(basename -- "$line")

    # get just the file name
    filename="${filenameEXT%.*}"

    # get the array ID
    arrIN2=(${filename//-/ })
    runID=(${arrIN2[-1]})
    echo $filenameEXT
    IDS_TO_RUN+=($runID)
  fi
done

# Print out comma-separated line of all runs that didn't save results.
printf -v joined '%s,' "${IDS_TO_RUN[@]}"
echo "${joined%,}"

