import torch
import numpy as np
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils

import optuna
import joblib

def time_objective(trial):
    
    #reading data
    transformed_datapath = "/home/pwendlan/tecde_contin_dec_dynamic/data_dict.p"
    pickle_map = read_from_file(transformed_datapath)
    training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)
    
    #treatment_options = 2
    
    hidden_channels = 15
    batch_size = 250
    hidden_states = 210
    lr = 0.008531767844627374
    activation = 'leakyrelu'
    num_depth = 2
    pred_act=None
    pred_states=None
    pred_depth=None
    
    thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
    treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])    
    
    model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=False, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth)
    model.load_state_dict(torch.load('/home/pwendlan/tecde_contin_dec_dynamic/final_enc.pth'))
    model=model.to(model.device)
    
    if training_processed is not None:
        train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
    if validation_processed is not None:
        validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)
 
    offset=5
    max_horizon=2
    
    input_channels_dec=3
    output_channels=2
    
    hidden_channels_dec = trial.suggest_int('hidden_channels_dec',1,30)
    batch_size_dec=trial.suggest_categorical('batch_size_dec',[16,32,64,125,250,500,1000])
    hidden_states_dec = trial.suggest_int('hidden_states_dec',16,1000)
    lr_dec = trial.suggest_uniform('lr_dec',0.0001,0.01)
    activation_dec = trial.suggest_categorical('activation_dec',['leakyrelu','tanh','sigmoid','identity'])
    num_depth_dec = trial.suggest_int('numdepth_dec',1,20)
    
    pred_act_dec = trial.suggest_categorical('pred_act_dec',['leakyrelu','tanh','sigmoid','identity'])
    pred_states_dec = trial.suggest_int('pred_states_dec',16,1000)
    pred_depth_dec = 1
    
    model_decoder = utils.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=hidden_channels,activation=activation_dec,num_depth=num_depth_dec, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec)
    model_decoder=model_decoder.to(model_decoder.device)
    
    done=False
    tries = 0
    while done == False:
        if tries > 3:
            done = True
            break
        try:
            loss = utils.train_dec(model,model_decoder,train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries, validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, offset=offset, max_horizon=max_horizon, lr=lr_dec, batch_size=batch_size_dec, patience=10, delta=0.0001, max_epochs=1000, hypopt=True, a_loss="spearman")
            done=True
        except Exception as e:
            print(e)
            tries = tries + 1
            loss = np.nan
            
    print(trial.number)
    print(loss)
    print(trial.params)
    
    torch.save(model_decoder.state_dict(), str("./final_model_hypopt" + str(trial.number) + ".pth"))

    return loss
    
study = optuna.create_study()
study.enqueue_trial({'hidden_channels_dec': 30, 'batch_size_dec': 1000, 'hidden_states_dec': 544, 'lr_dec': 0.006304913877025918, 'activation_dec': 'sigmoid', 'numdepth_dec': 10, 'pred_act_dec': 'leakyrelu', 'pred_states_dec': 923})
study.enqueue_trial({'hidden_channels_dec': 8, 'batch_size_dec': 1000, 'hidden_states_dec': 391, 'lr_dec': 0.00966781540535167, 'activation_dec': 'sigmoid', 'numdepth_dec': 20, 'pred_act_dec': 'tanh', 'pred_states_dec': 21})
study.enqueue_trial({'hidden_channels_dec': 4, 'batch_size_dec': 500, 'hidden_states_dec': 521, 'lr_dec': 0.004077174360596377, 'activation_dec': 'sigmoid', 'numdepth_dec': 9, 'pred_act_dec': 'identity', 'pred_states_dec': 680})
study.enqueue_trial({"hidden_channels_dec": 17, "batch_size_dec":64, "hidden_states_dec": 33, "lr_dec": 0.0050688746606452565, "activation_dec": 'tanh', "numdepth_dec": 15, "pred_act_dec": 'tanh', "pred_states_dec": 128})
study.enqueue_trial({"hidden_channels_dec": 20, "batch_size_dec":64, "hidden_states_dec": 804, "lr_dec": 0.000308314286773333, "activation_dec": 'sigmoid', "numdepth_dec": 1, "pred_act_dec": 'tanh', "pred_states_dec": 128})
study.enqueue_trial({"hidden_channels_dec": 22, "batch_size_dec":125, "hidden_states_dec": 802, "lr_dec": 0.0016227982436909543, "activation_dec": 'leakyrelu', "numdepth_dec": 13, "pred_act_dec": 'leakyrelu', "pred_states_dec": 798})
study.enqueue_trial({"hidden_channels_dec": 24, "batch_size_dec":125, "hidden_states_dec": 316, "lr_dec": 0.0010974778722903065, "activation_dec": 'leakyrelu', "numdepth_dec": 16, "pred_act_dec": 'leakyrelu', "pred_states_dec": 91, "preddepth": 3})

study.optimize(time_objective, n_trials=20, n_jobs=1)

print("Number of finished trials: ", len(study.trials))
    
print("Best trial:")
trial = study.best_trial

train_dir="/home/pwendlan/tecde_contin_dec_dynamic"

joblib.dump(study, "study_dec.pkl")


