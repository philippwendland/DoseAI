# %%

import torch
import numpy as np
import matplotlib.pyplot as plt
#import utils_paper_MimicIV as utils_paper
import pandas as pd
import pickle            
    
import torch
import numpy as np
# import utils_paper_new as utils_paper
# from  utils_paper_new import * 

# %%

import utils_paper_cancer as utils_paper
from  utils_paper_cancer import * 

import pandas as pd
import pickle

######
print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_seed_i = 2
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

save2 = 'C:/Users/Nutzer/Documents/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_semisynMimicIV/res/Trained_Decoder_seed_'+str(num_seed_i)+'_semisynMimicIV.pth'
save = 'C:/Users/Nutzer/Documents/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_semisynMimicIV/res/Trained_Encoder_seed_'+str(num_seed_i)+'_semisynMimicIV.pth'


#save2 = r'C:\Users\Nutzer\Documents\Paper_Code\Model_OptAB_DoseAI\Neutraining_Philipp\Trained_Decoder_seed_2_semisynmimic4.pth'
#save = r'C:\Users\Nutzer\Documents\Paper_Code\Model_OptAB_DoseAI\Neutraining_Philipp\Trained_Encoder_seed_0_semisynmimic4_encoder.pth'

path_data = "C:/Users/Nutzer/Documents/Paper_Code/SemiSynMimicIV/"

folder_h= 'C:/Users/Nutzer/Documents/Paper_Code'


#data
with open(path_data + "x_validate_syn.pkl", "rb") as datei:
    x_validate = pickle.load(datei)        
with open(path_data + "x_train_syn.pkl", "rb") as datei:
    x_train = pickle.load(datei)    
with open(path_data + "x_test_syn.pkl", "rb") as datei:
    x_test = pickle.load(datei)    
    
# Steuerung u    
with open(path_data + "u_validate_syn.pkl", "rb") as datei:
    u_validate = pickle.load(datei)
with open(path_data + "u_train_syn.pkl", "rb") as datei:
    u_train = pickle.load(datei)
with open(path_data + "u_test_syn.pkl", "rb") as datei:
    u_test = pickle.load(datei)    
    
    
with open(path_data + "t_validate.pkl", "rb") as datei:
     t_validate = pickle.load(datei)
with open(path_data + "t_train.pkl", "rb") as datei:
    t_train = pickle.load(datei)
with open(path_data + "t_test.pkl", "rb") as datei:
    t_test = pickle.load(datei)

# Prepare tensors
# Prepare base tensors
data_X_train = x_train[1:].transpose(0,1).to(device)
data_treatment_train = u_train.transpose(0,1).to(device)
data_active_train = ~x_train.transpose(0,1).isnan().to(device)
data_time_train = t_train[:-1,:,0:1].transpose(0,1).to(device)
data_toxic_train = None

data_X_test = x_validate[1:].transpose(0,1).to(device)
data_treatment_test = u_validate.transpose(0,1).to(device)
data_active_test = ~x_validate.transpose(0,1).isnan().to(device)
data_time_test = t_validate[:-1,:,0:1].transpose(0,1).to(device)
data_toxic_test = None

# Prepare covariables with time data
data_covariables_train = torch.cat([x_train,u_train,t_train[...,0:1]],dim=-1)[:-1].transpose(0,1).to(device)
data_covariables_test = torch.cat([x_validate,u_validate,t_validate[...,0:1]],dim=-1)[:-1].transpose(0,1).to(device)

# Forward fill the time column in covariables
time_idx = -1  # time is the last column
for data in [data_covariables_train, data_covariables_test]:
    # Get the time column
    time_col = data[..., time_idx]
    
    # Create a mask for NaN values
    mask = torch.isnan(time_col)
    
    # Forward fill NaN values
    valid_indices = torch.arange(time_col.shape[1], device=device)[None, :]
    valid_indices = valid_indices.expand(time_col.shape[0], -1)
    valid_indices[mask] = -1
    
    # Get the last valid index for each position
    last_valid = torch.zeros_like(valid_indices)
    for i in range(valid_indices.shape[1]):
        if i > 0:
            last_valid[:, i] = torch.where(
                valid_indices[:, i] == -1,
                last_valid[:, i-1],
                valid_indices[:, i]
            )
        else:
            last_valid[:, i] = torch.where(
                valid_indices[:, i] == -1,
                valid_indices[:, i],
                valid_indices[:, i]
            )
    
    # Fill NaN values with the last valid value
    filled_time = time_col.clone()
    for i in range(time_col.shape[0]):
        valid_vals = time_col[i, ~mask[i]]
        if len(valid_vals) > 0:
            filled_time[i, mask[i]] = valid_vals[-1]
    
    # Update the time column in the original tensor
    data[..., time_idx] = filled_time

print("\nData preparation checks:")
print("Training data time column NaNs:", torch.isnan(data_covariables_train[..., -1]).any())
print("Test data time column NaNs:", torch.isnan(data_covariables_test[..., -1]).any())
print("Training data shape:", data_covariables_train.shape)
print("Test data shape:", data_covariables_test.shape)

print("\nData preparation checks:")
print("Training data NaNs:", torch.isnan(data_covariables_train).any())
print("Test data NaNs:", torch.isnan(data_covariables_test).any())
print("Time column (train) NaNs:", torch.isnan(data_covariables_train[..., -1]).any())
print("Time column (test) NaNs:", torch.isnan(data_covariables_test[..., -1]).any())



print("\nData preparation checks:")
print("Training data NaNs:", torch.isnan(data_covariables_train).any())
print("Test data NaNs:", torch.isnan(data_covariables_test).any())
print("Time column (train) NaNs:", torch.isnan(data_covariables_train[..., -1]).any())
print("Time column (test) NaNs:", torch.isnan(data_covariables_test[..., -1]).any())

# # hyperparameters of the Encoder trail 19
hidden_channels = 7
batch_size = 1000
hidden_states = 174
lr = 0.0015972840572993194
activation = 'leakyrelu'
num_depth = 14
pred_act = 'leakyrelu'
pred_states = 275
pred_depth = 4
pred_comp=True


# # Hyperparameters of Decoder trail 15
hidden_channels_dec = 27
batch_size_dec = 1000
hidden_states_dec = 974
lr_dec = 0.002392251517863519
activation_dec = 'tanh'
num_depth_dec = 20
pred_act_dec = 'leakyrelu'
pred_states_dec = 203
pred_depth_dec = 2

#Hyperparameters of Decoder trail 1
#hidden_channels_dec = 25
#batch_size_dec = 1000
#hidden_states_dec = 825
#lr_dec = 0.0006940775886326423
#activation_dec = 'identity'
#num_depth_dec = 9
#pred_act_dec = 'leakyrelu'
#pred_states_dec = 403
#pred_depth_dec = 2

offset=0
rectilinear_index=0

data_thresh = torch.zeros(1)
model = utils_paper.NeuralCDE(input_channels=data_covariables_train.size(-1),
                              hidden_channels=hidden_channels,
                              hidden_states=hidden_states,
                              output_channels=1,
                              treatment_options=data_treatment_train.size(-1),
                              activation = activation,
                              num_depth=num_depth,
                              interpolation="linear",
                              pos=True,
                              thresh=data_thresh,
                              pred_comp=pred_comp,
                              pred_act=pred_act,
                              pred_states=pred_states,
                              pred_depth=pred_depth,
                              device=device)

# # Initializing and loading the Encoder
model.load_state_dict(torch.load(save, map_location=torch.device("cpu")))
model=model.to(device)


#########


z0_hidden_dimension_dec = hidden_channels+5

model_decoder = utils_paper.NeuralCDE(input_channels=1,
                                      hidden_channels=hidden_channels_dec,
                                      hidden_states=hidden_states_dec,
                                      output_channels=1,
                                      z0_dimension_dec=z0_hidden_dimension_dec,
                                      activation=activation_dec,
                                      num_depth=num_depth_dec,
                                      pos=True,
                                      thresh=data_thresh,
                                      pred_comp=True,
                                      pred_act=pred_act_dec,
                                      pred_states=pred_states_dec,
                                      pred_depth=pred_depth_dec,
                                      treatment_options=data_treatment_train.size(-1),
                                      device = device)#


model_decoder.load_state_dict(torch.load(save2, map_location=torch.device("cpu")))

model_decoder=model_decoder.to(device)

def plot_outcome_and_treatments_combined(data_X_test, predicted_outcomes, data_treatment_test, 
                                         patient_index=0, timepoints=10, folder_h='./'):
    """
    Plot actual & predicted outcomes (left y-axis) and both treatments (right y-axis)
    for a single patient in one combined figure.

    Args:
        data_X_test: Actual outcomes (ground truth)
        predicted_outcomes: Model predictions
        data_treatment_test: Treatment data (u1 and u2)
        patient_index: Index of the patient to plot
        timepoints: Number of timepoints to plot
        folder_h: Folder path for saving plots
    """
    # Move tensors to CPU and convert to numpy arrays
    data_X_test_np = data_X_test.cpu().numpy()
    predicted_outcomes_np = predicted_outcomes.detach().cpu().numpy()
    data_treatment_test_np = data_treatment_test.detach().cpu().numpy()

    time_points = np.arange(timepoints)

    # --- Dosisbereich berechnen ---
    dose_values = data_treatment_test_np[patient_index, :timepoints, :]
    dose_min = np.min(dose_values)
    dose_max = np.max(dose_values)
    dose_ylim = (dose_min - 0.2, dose_max + 0.2)

    # --- Kombinierter Plot ---
    fig, ax1 = plt.subplots(figsize=(6, 6))

    # Outcome (linke Y-Achse)
    ax1.plot(time_points, data_X_test_np[patient_index, :timepoints], 
             label='Actual Outcome', color='tab:blue', linewidth=2)
    ax1.plot(time_points, predicted_outcomes_np[patient_index, :timepoints], 
             label='Predicted Outcome', color='tab:orange', linewidth=2)
    ax1.set_xlabel('Time (τ)', fontsize=14, color='black')
    ax1.set_ylabel('Outcome', fontsize=14, color='black')
    ax1.tick_params(axis='x', labelsize=12, colors='black')
    ax1.tick_params(axis='y', labelsize=12, colors='black')
    ax1.set_ylim(np.min(data_X_test_np[patient_index, :timepoints])-2, 14)

    # Treatments (rechte Y-Achse)
    ax2 = ax1.twinx()
    ax2.scatter(time_points, data_treatment_test_np[patient_index, :timepoints, 0],
                label='Treatment u1', color='tab:green', s=15)
    ax2.scatter(time_points, data_treatment_test_np[patient_index, :timepoints, 1],
                label='Treatment u2', color='tab:red', s=15)
    ax2.set_ylabel('Dose', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelsize=12, colors='black')
    ax2.set_ylim(dose_ylim)

    # Legenden kombinieren
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{folder_h}\combined_patient_{patient_index}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
max_horizon =23
  
offset = 2
  
# Get predictions for test data
predicted_outcomes, _, _, _, _ = utils_paper.predict_decoder(
    model=model,
    model_decoder=model_decoder,
    validation_output=data_X_test,
    validation_toxic=data_toxic_test,
    validation_treatments=data_treatment_test,
    covariables=data_covariables_test,
    time_covariates=data_time_test,
    active_entries=data_active_test,
    static=None,
    offset=offset,
    max_horizon=max_horizon,
    dec_expand=False,
    med_dec=False,
    med_dec_start=True
)

# Create the plots

for i in range(data.shape[1]):
    print(i)
    plot_outcome_and_treatments_combined(data_X_test, predicted_outcomes, data_treatment_test, patient_index=i, timepoints=max_horizon, folder_h=folder_h)


asdf

# 1, 23, 25

# seed 0 sehr schlecht
# seed 1 doof
# seed 2 ziemlich gut

# seed 3 hat bisschen nichtlin.
# seed 4 besser

# seed 5 geht so

save_res = False
save_heat = False




unscaled=False
offset=0
max_horizon=23
invert = True
colorbar = True


#model.load_state_dict(torch.load(save, map_location=torch.device("cpu")))
#model_decoder.load_state_dict(torch.load(save2, map_location=torch.device("cpu")))


res_dic0= utils_paper.heatmap_pred_dec(model,
                                       model_decoder,
                                       offset=offset,
                                       max_horizon=max_horizon,
                                       loss='rmse',
                                       unscaled=unscaled,
                                       validation_output=data_X_test,
                                       validation_toxic=data_toxic_test,
                                       validation_treatments=data_treatment_test,
                                       covariables=data_covariables_test,
                                       time_covariates=data_time_test,
                                       active_entries=data_active_test,
                                       rectilinear_index=rectilinear_index,
                                       step=None,
                                       dec_expand=False,
                                       med_dec=False,
                                       med_dec_start=True,
                                       save_link=None,
                                       load_map=None,
                                       title=' Value 0',
                                       index=0,
                                       colorbar=colorbar,
                                       invert=invert)

