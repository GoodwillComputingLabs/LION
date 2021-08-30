#!/bin/bash -l
#SBATCH --time 06:00:00
#SBATCH --nodes 1
#SBATCH --job-name io_variability
#SBATCH --constraint=haswell
#SBATCH --qos=regular

export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled

module load python
pip install pyarrow
pip install -U memory_profiler

WORK_PATH='/global/homes/e/emily/eju/LION/cori_tests/test1'
rm -rf $WORK_PATH
mkdir $WORK_PATH
cp ~/eju/project_incite/io_variability_scripts/clustering.py $WORK_PATH
cp ~/eju/LION/cori_tests/run.py $WORK_PATH
cd $WORK_PATH

python3 -W ignore run.py
