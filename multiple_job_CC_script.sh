#!bin/bash
for c_init in {1.25,2,3,4.5}; do
    for num_simulations in {120,150,180,200}; do
        sbatch CC_script.sh $c_init $num_simulations
    done
done