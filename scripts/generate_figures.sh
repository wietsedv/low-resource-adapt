# commands to create all paper figures

rm -rf data/plots

# full data
# bash scripts/evaluate.sh
bash scripts/show_results.sh v3-noemb_v2 --select-mlm -l frisian  # manual copy to config-best.yml
bash scripts/show_results.sh v3-noemb_v2 --select-mlm -l groningen  # manual copy to config-best.yml
bash scripts/evaluate.sh -c best -l frisian --test
bash scripts/evaluate.sh -c best -l groningen --test
bash scripts/show_results.sh v3-noemb_v2 -c best --plot --latex

# data subsets
# bash scripts/evaluate.sh -c subsets

# bash scripts/show_results.sh v3-noemb_v3 -c subsets --select-mlm -l frisian  # manual copy to config-subsets-frisian.yml
# bash scripts/evaluate.sh -c subsets-frisian -l frisian --test
bash scripts/show_results.sh v3-noemb_v3 -c subsets-frisian --test --data-avg --latex

# bash scripts/show_results.sh v3-noemb_v3 -c subsets --select-mlm -l groningen  # manual copy to config-subsets-groningen.yml
bash scripts/show_results.sh v3-noemb_v3 -c subsets-groningen --plot-data --latex
