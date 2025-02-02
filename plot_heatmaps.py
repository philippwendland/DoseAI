import torch
import matplotlib.pyplot as plt
import utils_paper_cancer as utils

import src.utils.data_utils as pu


def main():
    
    #reading data
    transformed_datapath = "C:/Users/wendland/Documents/GitHub/TE-CDE-main/data_dict.p"
    pickle_map = pu.read_from_file(transformed_datapath)
    training_processed, validation_processed, test_processed = pu.process_data(pickle_map,toxicity=True,continuous=True)
    
    thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
    treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])
    
    #Hyperparameters
    
    #compepth3 zweiter 0.0918
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
    
    #training
    
    if training_processed is not None:
        train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
    if validation_processed is not None:
        validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)
    if test_processed is not None:
        test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test = utils.prep_map(test_processed,model.device)
    
    model.load_state_dict(torch.load('C:/Users/wendland/Documents/GitHub/final_model_enc.pth',map_location=torch.device('cpu')))
    
    #utils.train(model, train_X, train_toxic, train_treatments, covariables_x, active_entries, validation_X, validation_toxic, validation_treatments, covariables_x_val, active_entries_val, lr=lr, batch_size=batch_size, patience=10, delta=0.0001, max_epochs=1000)
    #pred_X_val, pred_a_val, loss_a_val, loss_output_val, loss_toxic_val = utils.predict(model, test_X, test_toxic, test_treatments, covariables_x_test, active_entries_test, a_loss='spearman')    
    
    # hyperparameters of decoder
    hidden_channels_dec = 22
    batch_size_dec = 125
    hidden_states_dec = 802
    lr_dec = 0.0016227982436909543
    activation_dec = 'leakyrelu'
    num_depth_dec = 13
    pred_act_dec = 'leakyrelu'
    pred_states_dec = 798
    pred_depth_dec = 1

    offset=0
    input_channels_dec=3  # Control = treatment_options + time
    #input_channels_dec=treatment_options+1
    
    output_channels=2
  
    # maximum prediction horizon  
    max_horizon=10
    
    model_decoder = utils.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=hidden_channels,activation=activation_dec,num_depth=num_depth_dec, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec)
    model_decoder=model_decoder.to(model_decoder.device)
    
    model_decoder.load_state_dict(torch.load('C:/Users/wendland/Documents/GitHub/final_model_dec.pth',map_location=torch.device('cpu')))
    
    #utils.train_dec_offset(model,model_decoder, train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries, validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, static,static_val, offset=offset, max_horizon=max_horizon, lr=0.001, batch_size=500, patience=10, delta=0.0001, max_epochs=1000)
    
    pred_X_val, pred_a_val, loss_a_val, loss_output_val, loss_toxic_val = utils.predict_decoder(model, model_decoder, test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, offset=offset, max_horizon=max_horizon, a_loss='spearman',static=None)
    
    # import matplotlib
    # matplotlib.rcParams.update({'font.size': 5})
    # fig, ax = plt.subplots(figsize=(1.5,2),dpi=400)
    # heatmap=ax.pcolor([[0,0.15]],cmap=plt.cm.seismic)
    # ax.set_visible(False)
    # fig.colorbar(heatmap)
    # plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/bilder_unscaled_dec_/colorbar.png')
    
    offset=0
    max_horizon=12
    
    title = 'Weight '
    load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/tmse_12_tox.pkl'
    utils.heatmap_pred_dec(model, model_decoder, validation_processed,validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, index=1, offset=offset, max_horizon=max_horizon, loss='mse',load_map=load_map,vmin=0,invert=True, title=title)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/t12_step_tox_mse.png', bbox_inches='tight')
    
    load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/tmse_12_tox.pkl'
    utils.heatmap_pred_dec(model, model_decoder, validation_processed,validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, index=1, offset=offset, max_horizon=max_horizon, loss='mse',load_map=load_map,vmin=0,vmax=1,invert=True,colorbar=False, title=title)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/t12_step_tox_mse_vmax.png', bbox_inches='tight')

    load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/tmse_12_tox.pkl'
    utils.heatmap_pred_dec(model, model_decoder, validation_processed,validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, index=1, offset=offset, max_horizon=max_horizon, loss='mse',load_map=load_map,vmin=0,vmax=0.15,invert=True,colorbar=False, title=title)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/t12_step_tox_mse_vmax015.png', bbox_inches='tight')
    
    title = 'Tumor Volume '
    load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/tmse_12.pkl'
    utils.heatmap_pred_dec(model, model_decoder, validation_processed,validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, index=0, offset=offset, max_horizon=max_horizon, loss='mse',load_map=load_map,vmin=0,invert=True, title=title)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/t12_step_mse.png', bbox_inches='tight')
    
    load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/tmse_12.pkl'
    utils.heatmap_pred_dec(model, model_decoder, validation_processed,validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, index=0, offset=offset, max_horizon=max_horizon, loss='mse',load_map=load_map,vmin=0,vmax=1,invert=True,colorbar=False, title=title)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/t12_step_mse_vmax.png', bbox_inches='tight')

    load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/tmse_12.pkl'
    utils.heatmap_pred_dec(model, model_decoder, validation_processed,validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val, index=0, offset=offset, max_horizon=max_horizon, loss='mse',load_map=load_map,vmin=0,vmax=0.15,invert=True,colorbar=False, title=title)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/t12_step_mse_vmax015.png', bbox_inches='tight')
    
    #load_map = 'c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/rmse.pkl'
    utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, index=1, offset=offset, max_horizon=max_horizon, loss='rmse',load_map=load_map)
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/20_step_tox_rmse.png', bbox_inches='tight')
    utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, index=1, offset=offset, max_horizon=max_horizon, loss='mae')
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/20_step_tox_mae.png', bbox_inches='tight')
    utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, index=1, offset=offset, max_horizon=max_horizon, loss='wape')
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/20_step_tox_wape.png', bbox_inches='tight')
    utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, index=1, offset=offset, max_horizon=max_horizon, loss='nrmse')
    plt.savefig('c:/users/wendland/documents/github/te-cde-main/bilder_unscaled_dec_/20_step_tox_nrmse.png', bbox_inches='tight')
