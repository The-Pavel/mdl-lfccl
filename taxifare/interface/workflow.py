from taxifare.interface.main import preprocess, evaluate, train
import os

from prefect import task, Flow

# Retrieve fresh data
@task
def preprocess_data(experiment_name):
    preprocess()
    preprocess(source_type='val')
    return True

# Evaluate past model performance
@task
def evaluate_old_model(status):
    mae = evaluate()
    return mae

# Train model
@task
def train_new_model(status):
    mae = train()
    return mae

# Put model in production
@task(log_stdout=True)
def notify_engineer(old_model_mae, new_model_mae):
    print(f"""
          Evaluation and Training results 📝\n
          Current Model MAE: {old_model_mae}\n
          Retrained MAE: {new_model_mae}
          """)

def build_flow():

    with Flow(name="wagonwab taxifare workflow") as flow:

        experiment_name = os.getenv("MLFLOW_EXPERIMENT")

        preproc_done = preprocess_data(experiment_name)

        old_mae = evaluate_old_model(preproc_done)
        new_val_mae = train_new_model(preproc_done)

        notify_engineer(old_mae, new_val_mae)

        return flow

if __name__ == '__main__':
    flow = build_flow()

    # flow.visualize()                   # visualize the DAG

    # flow.run()                         # local run
    flow.register("taxifare_project")
