#!/bin/bash

declare -a files=(
  'plot_merged_sim01.py'
  'plot_merged_sim02.py'
  'plot_merged_sim03.py'
  'plot_merged_sim04.py'
  'plot_merged_sim05.py'
  'plot_merged_sim06.py'
  'plot_merged_sim07.py'
  'plot_merged_sim08.py'
  'plot_tron_merged_beta.py'
  'plot_tron_merged_theta.py'
  # 'plot_tron_q_assist_sim.py'
)

for file in "${files[@]}"
do
  echo -e "Running file $file..."
  python $file
done
