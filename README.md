<!-- the repo is forked from https://github.com/seawee1/efficientalphazero -->

# Introduction

In order to do the experiments faster, we need to get everything work on Compute Canada (noted as CC), the cluster i'm using now is Beluga.  

The setup composes of 2 parts: VTR and python project code. These two parts will be covered below.  

Say now you are at /scratch on CC.

# Virtualenv
1. ```module load StdEnv/2020``` \
    ```module load python/3.10``` \
    ```module load cuda``` \
    ```module load clang```

2. ```virtualenv $your_env_name```

# VTR
VTR is the simulated environment to do chip design, but the only thing we need from it is:
Given a placement result produced by our RL agent, let it do the routing part and give us back some numerical results (wirelength/critical path delay), we use these results to compose an RL reward and use it for training.


1. ```git clone https://github.com/verilog-to-routing/vtr-verilog-to-routing.git```

2. cd into the repo

3. build VTR following the documentations: https://docs.verilogtorouting.org/en/latest/quickstart/


# RL-FPGA

1. ```git clone https://github.com/IRLL/RL-FPGA.git```

2. ```source $your_env_name/bin/activate```

2. cd into the repo

3. ```pip install -r requirements_cc.txt ```\
```pip install --no-index ray[default] ```\
```pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121```

4. open ~/.bashrc and add following lines\
```export VTR_ROOT=$your_root_path_of_VTR/vtr-verilog-to-routing```\
```export CUBLAS_WORKSPACE_CONFIG=:4096:8```\
```export WANDB_MODE=offline # log offline```\
 then, activate the changes ```source ~/.bashrc```

5. login wandb and paste your API key in configuratation. ```wandb login --relogin```

<!-- 6. before trying "sbatch job.sh", change the comments in that file! Then check whether you can finish the job without error in /slurm_out/%A.out file, the path I used to log result is this:
/home/qianxi/scratch/EDA/slurm_out/%A.out.  -->

<!-- # Usage
1. Compute Canada is probably not the recommended way to test your new feature, I'd suggest use your local machine (if you have one with gpu and the repo setup) to run your code for 1 time to verify the correctness of your code first, and get an estimate of time usage so that you know how much time you should ask for when running on CC.

2. Then pull the code on CC, try to schedule 1 job first use the script mentioned below. Take a look at xxxx.out file at /home/qianxi/scratch/EDA/slurm_out and make sure it is finished correctly.

3. My folder structure looks like:

/scratch/  
/scratch/RL-in_FPGA  
/scratch/vtr-veri.....  
/scratch/experiment_results (I guess this is auto created but you can create an empty one first.)  
/scratch/slurm_out (auto generated)     -->
<!-- ```
0.
Important: You need to change things related to qianxi to yours, to do that, no need to modify config files, 
just open single_job_submitter.sh and multiple_jobs_trigger.sh, override vtr_root, eda_root, result_path with your paths.
Comment out some module load things and activate virtualenv things based on instrucitons inside each bash file (to run on local, not CC).
You also need to modify the comments at the beginning of these scripts (if you are not familiar with CC, these are task infos, to tell the scheduler what resources you need for the program.) 

1.
# On your local machine.
# to run on your local machine, Make sure you change the email/path and everything related to Qianxi Li before you run this. 

bash single_job_submitter.sh


2.
# On compute canada:
# to submit multiple jobs on compute canada. Make sure you change the email/path and everything related to Qianxi Li before you run this.

bash multiple_jobs_trigger.sh


3.
# On compute canada:
# to submit single job on compute canada. Make sure you change the email/path and everything related to Qianxi Li before you run this.

sbatch single_job_submitter.sh

4. After running
If you enabled wandb for logging (i.e. debug=False, which you should do so, otherwise you are not logging things), then on compute canada it will log everything locally (compute node no internet), you need to sync all wandb offline runs to their server.

4.1
After you finish some training, you should see a wandb folder inside /RL-in_FPGA, cd into it, then run this:

wandb sync --include-offline ./offline-*

4.2
In case /experiment_results, /slurm_out and /RL-in_FPGA/wandb is full, you may want to remove all the files inside manually, as long as you run the sync things in 4.1 then you shouldn't really need the things in the above three folders.
```




If you are having trouble installing torch and torch_geometric on CC, I recommend the following:
1) Delete your virtualenv and create a fresh one.
2) Activate the environment. Edit your req.txt and remove the lines that specify the torch version and torch_geometric version because you will be installing this manually. After removing those two libraries, go ahead and install the rest of the libraries using the regular "pip install -r req.txt" command
3) After all the libraries are installed, manually install torch and torch_geometric using these commands:
   a) pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
   b) pip install torch_geometric
5) Check the torch version (it should be 1.13 now). If you run into a "memory error" while pip installing either library, you can use the "--no-cache-dir" flag. 

 -->
