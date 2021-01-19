# Code for Experiments in CartPole Task

### Requirement

tensorflow==1.14.0



### Running Experiments

##### Preparation

```shell
# Collect dataset
python CartPole_Gene_Data.py --tau 1.0 --seed 100 200 300 400 500 --n-ros 50
python CartPole_Gene_Data.py --tau 1.0 --seed 100 200 300 400 500 --n-ros 100
python CartPole_Gene_Data.py --tau 1.0 --seed 100 200 300 400 500 --n-ros 200
python CartPole_Gene_Data.py --tau 1.0 --seed 100 200 300 400 500 --n-ros 500

# Estimate true value of behavior/target policy
python OnPolicy_Est.py --gamma 0.999 --tau 0.25 --ep-len 10000 --traj-num 150
python OnPolicy_Est.py --gamma 0.999 --tau 0.5 --ep-len 10000 --traj-num 150
python OnPolicy_Est.py --gamma 0.999 --tau 1.0 --ep-len 10000 --traj-num 150
python OnPolicy_Est.py --gamma 0.999 --tau 1.5 --ep-len 10000 --traj-num 150
python OnPolicy_Est.py --gamma 0.999 --tau 2.0 --ep-len 10000 --traj-num 150
```



##### Experiment 1 (Changing $\tau$ with fixed dataset size)

```shell
# run experiments (batch run)
python Run_MQL.py --tau 0.25 0.5 1.5 2.0 --dataset-seed 100 200 300 400 500 --n-ros 200
python Run_MWL.py --tau 0.25 0.5 1.5 2.0 --dataset-seed 100 200 300 400 500 --n-ros 200
python Run_DualDICE.py --tau 0.25 0.5 1.5 2.0 --dataset-seed 100 200 300 400 500 --n-ros 200
python Run_MSWL.py --tau 0.25 0.5 1.5 2.0 --dataset-seed 100 200 300 400 500 --n-ros 200

# plot
python plot_tau.py --tau 0.25 0.5 1.5 2.0 --est MQL MWL DualDICE MSWL
```



##### Experiment 2 (Change dataset size with fixed $\tau=1.5$)

```shell
# run experiments (batch run)
python Run_MQL.py --tau 1.5 --dataset-seed 100 200 300 400 500 --n-ros 50 100 200 500
python Run_MWL.py --tau 1.5 --dataset-seed 100 200 300 400 500 --n-ros 50 100 200 500
python Run_DualDICE.py --tau 1.5 --dataset-seed 100 200 300 400 500 --n-ros 50 100 200 500
python Run_MSWL.py --tau 1.5 --dataset-seed 100 200 300 400 500 --n-ros 50 100 200 500

# plot
python plot_sample_size.py --tau 1.5 --est MQL MWL DualDICE MSWL --n-ros 50 100 200 500
```

