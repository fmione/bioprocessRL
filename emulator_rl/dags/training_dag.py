import datetime as dt
import json
from airflow.models.dag import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from docker.types import Mount
from pathlib import Path

try:
    host_path = Variable.get("host_path", deserialize_json=True)
except:
    print("Host path has not been addded to the airflow UI variables or it has not been done correctly!")
    host_path = ""

remote_path = "/opt/airflow/dags"


def base_docker_node(task_id, command, retries=3, retry_delay=dt.timedelta(minutes=2), 
                     execution_timeout=dt.timedelta(minutes=10), trigger_rule='all_success'):

    return DockerOperator(
        task_id=task_id,
        image="training",
        auto_remove="force",
        working_dir=f"{remote_path}/scripts/training_dag",
        command=command,
        mounts=[Mount(source=str(host_path), target=remote_path, type='bind')],
        mount_tmp_dir=False,
        network_mode="bridge",
        retries=retries,
        retry_delay=retry_delay,
        execution_timeout=execution_timeout,
        trigger_rule=trigger_rule 
    )


with DAG(
        dag_id="Training_DAG",
        description="Kiwi experiment emulator.",
        start_date=dt.datetime.now(),
        schedule_interval=None,
        catchup=False,
        is_paused_upon_creation=True
) as dag:
    
    start = EmptyOperator(
        task_id="start"
    )

    last_node = start
    
    for  n_envs in [4, 8]:
        for lr in [0.0001, 0.001, 0.01]:
            for ec in [0.001, 0.01]:
                for cp in [True, False]:
                   
                    train = base_docker_node(
                        task_id=f"train_n_envs_{n_envs}_lr_{lr}_ec_{ec}_cp_{cp}",
                        command=f"python training.py {n_envs} {lr} {ec} {cp}",
                        trigger_rule="all_done"
                    )
                    
                    last_node >> train
                    last_node = train
   
    