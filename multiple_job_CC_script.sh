for c_init in {0.5,1.25,2,3,4.5}; do
    for num_simulations in {70,80,100}; do
        sbatch CC_script.sh $c_init $num_simulations
    done
done