#!/bin/zsh
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -p MISC-56c
#SBATCH --time 100:00:00

BASE="$HOME/PycharmProjects/rdn-wdp-python"

job="$SLURM_ARRAY_TASK_ID"
plist="$BASE/testing/params_sg.ssv"
params=$(sed -n "1p" $plist)
exe="$BASE/analysis/Command/cluster.py"

gene=$(echo $params | cut -d ';' -f 1)
suffix=$(echo $params | cut -d ';' -f 2)
sample_list=$(echo $params | cut -d ';' -f 3)

# Fixed parameters - copy this file and modify below
clusters='4,5,6,7,8,9,10'
samples='20'
repeats='1'
mode='classify'
outfile="clustering_${gene}_${suffix}_s${samples}r${repeats}"

outdir="$BASE/testing/jobs/666"

cmd="$exe -g $gene -l $sample_list -k $clusters -r $repeats -n $samples -d $mode --furrow --log debug --no-bad $BASE/testing/samples_complete.csv $outdir/$outfile"

mkdir -p $outdir
echo "${cmd}"
${=cmd}
