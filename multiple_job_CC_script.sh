#!bin/bash
for c_init in {1.25,2,2.5,3,3.5,4.5}; do
    for num_simulations in {150,180,200,220,240}; do
        sbatch CC_script.sh $c_init $num_simulations
    done
done