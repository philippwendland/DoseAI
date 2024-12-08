import numpy as np
import torch
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer
import pymoo_utils_cancer as pu
import pickle

path='/work/wendland/pymoo_2500_discrete/'

transformed_datapath = path+"data_dict.p"

pickle_map = read_from_file(transformed_datapath)
training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)

thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])

scaling_params=pickle_map["scaling_data"]

treatment_options = 2

#compdepth3 
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

model = utils_paper_cancer.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth, device='cpu')
#model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth, device='cpu')
model=model.to(model.device)

model.load_state_dict(torch.load(path+'final_model_enc.pth',map_location=torch.device('cpu')))

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

model_decoder = utils_paper_cancer.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=hidden_channels,activation=activation_dec,num_depth=num_depth_dec, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec, device='cpu')
model_decoder=model_decoder.to(model_decoder.device)

model_decoder.load_state_dict(torch.load(path+'final_model_dec.pth',map_location=torch.device('cpu')))

num_patients=1000

offset=0
max_time=6
treat1_list= [0,3,5,7]
treat2_list=[0,1,2,3]

alg='optuna'
n_gen=2500

delta_list = [6,9,12,15,18]
BIGGER_SIZE=40

CDE_opt_trt_in_real_model_tens = np.empty(shape=[num_patients,len(delta_list),3])
CDE_opt_trt_in_real_model_tens[:] = np.nan

Real_opt_trt_progression_tens = np.empty(shape=[num_patients,len(delta_list),3])
Real_opt_trt_progression_tens[:] = np.nan

CDE_opt_trt_dict = {}
real_opt_trt_dict = {}

CDE_opt_trt_in_real_model_compl = {}
Real_opt_trt_progression_compl = {}

# Indicating whether discrete/categorical treatment dosages or continuous treatment dosages for treatment optimization
discrete_treatment=True

# i and patient describe patient number
for i in range(1001):
    print(i)
    seed=i
    plt_save=True
    
    CDE_opt_trt_in_real_model, Real_opt_trt_progression, CDE_opt_trt, real_opt_trt, CDE_opt_trt_in_real_model_complete,Real_opt_trt_progression_complete = pu.plot_individual_patients(model, model_decoder, treat1_list=treat1_list, treat2_list=treat2_list, delta_list = delta_list, seed=seed, offset=offset, max_time=max_time, scaling_params=scaling_params, alg=alg, n_gen=n_gen, plot_size=BIGGER_SIZE, plt_save=plt_save, path=path,discrete_treatment=discrete_treatment)
    
    CDE_opt_trt_in_real_model_tens[i] = CDE_opt_trt_in_real_model
    Real_opt_trt_progression_tens[i] = Real_opt_trt_progression
    CDE_opt_trt_dict[i] = CDE_opt_trt
    real_opt_trt_dict[i] = real_opt_trt
    CDE_opt_trt_in_real_model_compl[i] = CDE_opt_trt_in_real_model_complete
    Real_opt_trt_progression_compl[i] = Real_opt_trt_progression_complete

    ylim=None
    pu.plot_individual_patients(model, model_decoder, treat1_list=treat1_list, treat2_list=treat2_list, delta_list = delta_list, seed=seed, offset=offset, max_time=max_time, scaling_params=scaling_params, alg=alg, n_gen=n_gen, plot_size=BIGGER_SIZE, plt_save=plt_save, sim=False, CDE_opt_trt_in_real_model=CDE_opt_trt_in_real_model, Real_opt_trt_progression=Real_opt_trt_progression, opt_treatment_multi = CDE_opt_trt,  real_opt_treatment_multi = real_opt_trt, legend=True, path=path, ylims=ylim, discrete_treatment=discrete_treatment)

    np.save(path+'out.npy',CDE_opt_trt_in_real_model_tens)
    np.save(path+'out_real.npy',Real_opt_trt_progression_tens)  
    
    with open(path+'out_complete.pkl', 'wb') as f:
        pickle.dump(CDE_opt_trt_in_real_model_compl, f)
    
    with open(path+'out_real_complete.pkl', 'wb') as f:
        pickle.dump(Real_opt_trt_progression_compl, f)
    
    with open(path+'res.pkl', 'wb') as f:
        pickle.dump(CDE_opt_trt_dict, f)
        
    with open(path+'res_real.pkl', 'wb') as f:
        pickle.dump(real_opt_trt_dict, f)
        
