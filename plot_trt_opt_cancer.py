import numpy as np
import torch
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils
import pymoo_utils_cancer as pu
import pickle

transformed_datapath = "C:/Users/wendland/Documents/GitHub/TE-CDE-main/data_dict.p"
pickle_map = read_from_file(transformed_datapath)
training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)

thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])

scaling_params=pickle_map["scaling_data"]

treatment_options = 2


#hyperparameters, compdepth3 
hidden_channels = 16
batch_size = 250
hidden_states = 578
lr = 0.004239690693777566
activation = 'leakyrelu'
num_depth = 2
pred_act = 'tanh'
pred_states = 128
pred_depth = 4
pred_comp=True

model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth)
model=model.to(model.device)

model.load_state_dict(torch.load('C:/Users/wendland/Documents/GitHub/final_model_enc.pth',map_location=torch.device('cpu')))

input_channels_dec=3
output_channels=2

hidden_channels_dec = 22
batch_size_dec = 125
hidden_states_dec = 802
lr_dec = 0.0016227982436909543
activation_dec = 'leakyrelu'
num_depth_dec = 13
pred_act_dec = 'leakyrelu'
pred_states_dec = 798
pred_depth_dec = 1

model_decoder = utils.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=hidden_channels,activation=activation_dec,num_depth=num_depth_dec, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec)
model_decoder=model_decoder.to(model_decoder.device)

model_decoder.load_state_dict(torch.load('C:/Users/wendland/Documents/GitHub/final_model_dec.pth',map_location=torch.device('cpu')))

for i in range(0,18):
    
    path_pymoo_multi = "C:/Users/wendland/Documents/Cancer/pymoo_server/pymoo_5000_discrete/"
    path_pymoo_multi = "C:/Users/wendland/Documents/Cancer/pymoo_server/pymoo_250_discrete/"
    
    out_multi = np.load(path_pymoo_multi+'out.npy')
    out_real_multi = np.load(path_pymoo_multi+'out_real.npy')

    with open(path_pymoo_multi + 'res.pkl', 'rb') as f:
        opt_treatment_multi = pickle.load(f)

    with open(path_pymoo_multi + 'res_real.pkl', 'rb') as f:
        real_opt_treatment_multi = pickle.load(f)
        
    with open(path_pymoo_multi + 'out_complete.pkl', 'rb') as f:
        out_multi_complete = pickle.load(f)

    with open(path_pymoo_multi + 'out_real_complete.pkl', 'rb') as f:
        out_real_multi_complete= pickle.load(f)
    
    patient=i
    
    offset=0
    max_time=6
    treat1_list= [0,3,5,7]
    treat2_list=[0,1,2,3]
    alg='optuna'
    n_gen=2500
    delta_list = [6,9,12,15,18]
    plot_size=40
    
    plt_save=True
    if i==6:
        ylim=[0,100]
    elif i==16:
        ylim=[0,50]
    elif i==28:
        ylim=[0,130]
    else:
        ylim=None
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/trtopt/discrete5000_'
    
    pu.plot_individual_patients(model, model_decoder, treat1_list=treat1_list, treat2_list=treat2_list, delta_list = delta_list, seed=patient, offset=offset, max_time=max_time, scaling_params=scaling_params, alg=alg, n_gen=n_gen, plot_size=plot_size, plt_save=plt_save, sim=False, CDE_opt_trt_in_real_model=out_multi[patient], Real_opt_trt_progression=out_real_multi[patient], opt_treatment_multi = opt_treatment_multi[patient], real_opt_treatment_multi = real_opt_treatment_multi[patient], legend=True, path=path, ylims=ylim)

    # Example data extraction based on provided structure
    cancer_volumes = {
        6: out_multi_complete[patient][6]['cancer_volume'][0],
        9: out_multi_complete[patient][9]['cancer_volume'][0],
        12: out_multi_complete[patient][12]['cancer_volume'][0],
        15: out_multi_complete[patient][15]['cancer_volume'][0],
        18: out_multi_complete[patient][18]['cancer_volume'][0]
    }
    
    chemo_doses = {
        6: opt_treatment_multi[patient][6][:, 0],  # Exclude the last time point
        9: opt_treatment_multi[patient][9][:, 0],
        12: opt_treatment_multi[patient][12][:, 0],
        15: opt_treatment_multi[patient][15][:, 0],
        18: opt_treatment_multi[patient][18][:, 0]
    }
    
    radio_doses = {
        6: opt_treatment_multi[patient][6][:, 1],  # Exclude the last time point
        9: opt_treatment_multi[patient][9][:, 1],
        12: opt_treatment_multi[patient][12][:, 1],
        15: opt_treatment_multi[patient][15][:, 1],
        18: opt_treatment_multi[patient][18][:, 1]
    }
    
    if i==6:
        ylim = [0,130]
    elif i==16:
        ylim= [0,65]
    elif i==28:
        ylim=[30,140]
    else:
        ylim = None
    # Call the function to plot the data with specified text and label sizes
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/trtopt/discrete5000_patient' + str(patient) + '_cancer'
    
    pu.plot_dynamic(cancer_volumes, chemo_doses, radio_doses, ylim=ylim, plt_save=True,path=path)
    
    toxicity = {
        6: out_multi_complete[patient][6]['toxicity'][0],
        9: out_multi_complete[patient][9]['toxicity'][0],
        12: out_multi_complete[patient][12]['toxicity'][0],
        15: out_multi_complete[patient][15]['toxicity'][0],
        18: out_multi_complete[patient][18]['toxicity'][0]
    }
    
    if i==6:
        ylim = [100,125]
    elif i==16:
        ylim=[116,134]
    elif i==28:
        ylim=[63,80]
    else:
        ylim=None
    
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/trtopt/discrete5000_patient' + str(patient) + '_weight'
    
    if i==16:
        d=True
    else:
        d=False
    pu.plot_dynamic(toxicity, chemo_doses, radio_doses, metric='weight', ylim=ylim, plt_save=True,path=path,d=d)
    
    location_c=[0.025,0.1]
    location_r=[0.025,0.1]
    
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/doses/discrete5000_'+str(patient)+'_'
    
    pu.plot_dosages(chemo_doses=chemo_doses,radio_doses=radio_doses,metric='chemo',plt_save=True,path=path,legend=True,location=location_c)

    pu.plot_dosages(chemo_doses=chemo_doses,radio_doses=radio_doses,metric='radio',plt_save=True,path=path,legend=True,location=location_r)
    
    
    #path_pymoo_multi = "C:/Users/wendland/Documents/Cancer/pymoo_server/pymoo_5000_real/"
    path_pymoo_multi = "C:/Users/wendland/Documents/Cancer/pymoo_server/pymoo_5000_real/"
    
    out_multi = np.load(path_pymoo_multi+'out.npy') #out6
    out_real_multi = np.load(path_pymoo_multi+'out_real.npy') #out_real6

    with open(path_pymoo_multi + 'res.pkl', 'rb') as f:
        opt_treatment_multi = pickle.load(f)
    
    with open(path_pymoo_multi + 'res_real.pkl', 'rb') as f:
        real_opt_treatment_multi = pickle.load(f)
        
    with open(path_pymoo_multi + 'out_complete.pkl', 'rb') as f:
        out_multi_complete = pickle.load(f)
    
    with open(path_pymoo_multi + 'out_real_complete.pkl', 'rb') as f:
        out_real_multi_complete= pickle.load(f)
        
        
    offset=0
    max_time=6
    treat1_list= [0,3,5,7]
    treat2_list=[0,1,2,3]
    alg='optuna'
    n_gen=2500
    delta_list = [6,9,12,15,18]
    plot_size=40
    
    plt_save=True
    ylim=None
    
    if i==6:
        ylim=[0,100]
    elif i==16:
        ylim=[0,50]
    elif i==28:
        ylim=[0,130]
    else:
        ylim=None
    
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/trtopt/real5000_'
    
    pu.plot_individual_patients(model, model_decoder, treat1_list=treat1_list, treat2_list=treat2_list, delta_list = delta_list, seed=patient, offset=offset, max_time=max_time, scaling_params=scaling_params, alg=alg, n_gen=n_gen, plot_size=plot_size, plt_save=plt_save, sim=False, CDE_opt_trt_in_real_model=out_multi[patient], Real_opt_trt_progression=out_real_multi[patient], opt_treatment_multi = opt_treatment_multi[patient], real_opt_treatment_multi = real_opt_treatment_multi[patient], legend=True, path=path, ylims=ylim)
    
    # Example data extraction based on provided structure
    cancer_volumes = {
        6: out_multi_complete[patient][6]['cancer_volume'][0],
        9: out_multi_complete[patient][9]['cancer_volume'][0],
        12: out_multi_complete[patient][12]['cancer_volume'][0],
        15: out_multi_complete[patient][15]['cancer_volume'][0],
        18: out_multi_complete[patient][18]['cancer_volume'][0]
    }
    
    chemo_doses = {
        6: opt_treatment_multi[patient][6][:, 0],  # Exclude the last time point
        9: opt_treatment_multi[patient][9][:, 0],
        12: opt_treatment_multi[patient][12][:, 0],
        15: opt_treatment_multi[patient][15][:, 0],
        18: opt_treatment_multi[patient][18][:, 0]
    }
    
    radio_doses = {
        6: opt_treatment_multi[patient][6][:, 1],  # Exclude the last time point
        9: opt_treatment_multi[patient][9][:, 1],
        12: opt_treatment_multi[patient][12][:, 1],
        15: opt_treatment_multi[patient][15][:, 1],
        18: opt_treatment_multi[patient][18][:, 1]
    }
    
    # Call the function to plot the data with specified text and label sizes
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/trtopt/real5000_patient' + str(patient) + '_cancer'
    
    if i==6:
        ylim = [0,130]
    elif i==16:
        ylim= [0,65]
    elif i==28:
        ylim=[30,140]
    else:
        ylim = None
    
    pu.plot_dynamic(cancer_volumes, chemo_doses, radio_doses, ylim=ylim, plt_save=True,path=path)
    
    toxicity = {
        6: out_multi_complete[patient][6]['toxicity'][0],
        9: out_multi_complete[patient][9]['toxicity'][0],
        12: out_multi_complete[patient][12]['toxicity'][0],
        15: out_multi_complete[patient][15]['toxicity'][0],
        18: out_multi_complete[patient][18]['toxicity'][0]
    }
    
    if i==6:
        ylim = [100,125]
    elif i==16:
        ylim=[116,134]
    elif i==28:
        ylim=[63,80]
    else:
        ylim=None
        
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/trtopt/real5000_patient' + str(patient)
    
    pu.plot_dynamic(toxicity, chemo_doses, radio_doses, metric='weight', ylim=ylim,plt_save=True,path=path)
    
    location_c='upper right'
    location_r='upper right'
    
    path='C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_unscaled_dec_/doses/real5000_'+str(patient)+'_'
    
    pu.plot_dosages(chemo_doses=chemo_doses,radio_doses=radio_doses,metric='chemo',plt_save=True,path=path,legend=True)

    pu.plot_dosages(chemo_doses=chemo_doses,radio_doses=radio_doses,metric='radio',plt_save=True,path=path,legend=True)
    