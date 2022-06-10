#!/bin/bash

declare -a files=(
  'plot_sim01.py'
  'plot_sim02.py'
  'plot_sim03.py'
  'plot_sim04.py'
  'plot_sim05.py'
  'plot_sim06.py'
  'plot_sim07.py'
  'plot_sim08.py'
)

for file in "${files[@]}"
do
  echo -e "Running file $file..."
  python $file
done
