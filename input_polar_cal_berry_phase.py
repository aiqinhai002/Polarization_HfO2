#!python
#
from dwgen import polar_cal_berry_phase

in_file = 'HfO2_f.POSCAR'
base_file = 'HfO2_Base.POSCAR'
N = 21
exten_N = 3
geom = True
polar_dir = 'c'
vasp_run = True
run_str = 'mpirun -np 56 vasp_std'
polar_cal_berry_phase(in_file, base_file, N=N, exten_N=exten_N, geom=geom, polar_dir=polar_dir,
    vasp_run=vasp_run, run_str=run_str)
