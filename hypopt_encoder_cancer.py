import torch
import numpy as np
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils

import optuna
import joblib
import pandas as pd

def time_objective(trial):
    
    #reading data
    transformed_datapath = "/home/pwendlan/tecde_contin_dynamic/data_dict.p"
    pickle_map = read_from_file(transformed_datapath)
    training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)
    
    treatment_options = 2
    
    hidden_channels = trial.suggest_int('hidden_channels',1,30)
    batch_size=trial.suggest_categorical('batch_size',[16,32,64,125,250,500,1000])
    hidden_states = trial.suggest_int('hidden_states',16,1000)
    lr = trial.suggest_uniform('lr',0.0001,0.01)
    activation = trial.suggest_categorical('activation',['leakyrelu','tanh','sigmoid','identity'])
    num_depth = trial.suggest_int('numdepth',1,20)
    
    thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
    treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])    
    
    model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=treatment_options, activation = activation, num_depth=num_depth, interpolation="linear", continuous=True,treat_thresh=treat_thresh, pos=True, thresh=thresh)
    
    model=model.to(model.device)
    
    if training_processed is not None:
        train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
    if validation_processed is not None:
        validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)
    
    done=False
    tries = 0
    while done == False:
        if tries > 3:
            done = True
            break
        try:
            loss = utils.train(model, train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries, validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, lr=lr, batch_size=batch_size, patience=10, delta=0.0001, max_epochs=1000, hypopt=True, a_loss='spearman')
            
            done=True
        except Exception as e:
            print(e)
            tries = tries + 1
            loss = np.nan
            
    print(trial.number)
    print(loss)
    print(trial.params)
    
    torch.save(model.state_dict(), str("./final_model_hypopt" + str(trial.number) + ".pth"))

    return loss
    
study = optuna.create_study()

study.optimize(time_objective, n_trials=20, n_jobs=1)

print("Number of finished trials: ", len(study.trials))
    
print("Best trial:")
trial = study.best_trial

train_dir="/home/pwendlan/tecde_contin_dynamic"

joblib.dump(study, "study.pkl")


