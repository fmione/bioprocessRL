# Deep Reinforcement Learning for High-Throughput Bioprocess Development

Complete documentation and code to reproduce the results of the paper entitled: **"Online Redesign of Dynamic Experiments for High-Throughput Bioprocess Development using Deep Reinforcement Learning"**.

## Authors
Martin F. Luna $^a$, Federico M. Mione $^a$, Ernesto C. Martinez $^{a,b}$ and M. Nicolas Cruz Bournazou $^b$.

$^a$ *INGAR (CONICET - UTN). Avellaneda 3657, Santa Fe, Argentina*<br>
$^b$ *Technische Universität Berlin, Institute of Biotechnology, Chair of Bioprocess Engineering. Berlin, Germany*

## Abstract
For efficiency and reproducibility, modern biotech laboratories increasing rely on robotic platforms to perform complex, dynamic experiments to generate informative data for bioprocess development and knowledge discovery. Furthering the goal of self-driving biolabs imposes the need of automating cognitive demanding tasks such as redesigning online an experiment to maximize information gain in the face of different sources of uncertainty including microorganism behavior, sensor failure, intrinsic platform variability and modeling errors. In this work, a reinforcement learning (RL) based formulation of the online redesign problem for dynamic experiments is proposed. Simulation-based learning of a redesign policy for parallel cultivations in a high-throughput platform is discussed to provide implementation details regarding the RL agent design (perceptions and actions) and the reward function used. The simulated environment and a training workflow for sequential information control are integrated with the Proximal Policy Optimization (PPO) algorithm to learn how to modify 'on the fly' an offline design based solely on previous observations and actions in a given experiment. Results obtained demonstrate the feasibility of using RL to guarantee the information content of generated data and increase the level of automation that can be used in high-throughput platforms.

## Training reproducibility

### Required version

* Python >= 3.11.0
* Pip >= 25.0.0

### PPO Hyperparameters
Hyperparameter selection during training was performed using a grid search based on the Case Study 2 environment. The process can be reproduced by executing the following script: [/Case2/hyperparameters_train.py](Case2/hyperparameters_train.py)

### Case studies
To reproduce agent training for each case study, a [requirements.txt](requirements.txt) file is provided to create a virtual environment using Python venv. The steps are the following:

* Create the environment:

        python -m venv bioprocessRL

* Activate the environment:

        # Linux
        source bioprocessRL/bin/activate 
        
        # Windows
        .\bioprocessRL\Scripts\activate

* Install dependencies:

        pip install -r requirements.txt

* Go to the corresponding directory (Case1, Case2, Case3):

        cd Case1
 
* Perform the offline optimization:
 
        python Offline_optimization_Case1.py

* Excecute the training file:
 
        python Train_Case1.py

* Monitor the training process with Tensorboard:

        tensorboard --logdir=logs/final --port=6006

* Generate the plots :

        python Plots_Case1.py

* Finally, compare the 3 agents :
 
        python Model_comparisson_Case1.py


## Online emulation reproducibility

### Installation steps

To reproduce the emulator results, please follow these steps:

* Install [Docker](https://www.docker.com/).

* Navigate to the `emulator_rl` directory and set up the Apache Airflow service:

        docker-compose up -d airflow-init 

* Install all the remaining services:

        docker-compose up -d

* Once installation is complete (this may take several minutes), the Apache Airflow web interface will be accessible at: at http://localhost:8080/.
**Log in with user: airflow, and password: airflow.**

* Finally, Airflow variables must be set:

    * In the upper ribbon, navigate to **Admin** > **Variables**.
    * Click on **Choose a file**, and locate the `config.json` file within the `/dags` directory.
    * Once the JSON file is uploaded, click on **Import variables**.
    
    **IMPORTANT**: The **host_path variable must be changed** to the absolute local path where the `/dags` folder is located.


### Execution of DAGs
Two DAGs are defined: one for the emulator (*Emulator_2.0_DAG*) and the other for the agent-based computational control (*RL_Controller_DAG*). To execute both, the corresponding toggle switch must be activated, followed by pressing the play button for each DAG.

Case Study 1 is configured as the default. To reproduce other case studies, you must first create the configuration files for both, the controller and the emulator. For this purpose in `dags/scripts/emulator_dag/` you can find the `method_createDesign_caseX.py` file to create the `EMULATOR_config.json` neccesary to run the simulation. You should run it an place the resulting file in the same directory. For the controller, the corresponding folder is located at `dags/scripts/controller_dag/` with the `method_create_config_caseX.py` file.

The default simulation mode is accelerated, where 1 minute of real time represents 1 hour of experimental simulation, resulting in 14 minutes to complete the simulation.

### Monitoring tool
To monitor the simulated experiments, a tool using a Streamlit-based interface can be accessed at the following address: http://localhost:8501/

Please, select the simulated experiment with RUN ID: 623.

## License
This project is under an MIT license. See the LICENSE file for more details.
