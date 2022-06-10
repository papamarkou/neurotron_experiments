#!/bin/bash

declare -a files=(
  'run_sim01.py'
  'run_sim02.py'
  'run_sim03.py'
  'run_sim04.py'
  'run_sim05.py'
  'run_sim06.py'
  'run_sim07.py'
  'run_sim08.py'
)

for file in "${files[@]}"
do
  echo -e "Running file $file..."
  python $file
done
