
import torch
import pickle
import pandas as pd
import numpy as np
import sklearn
import torch
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from utils_MimicIV import RandomFourierFeaturesFunction, DiscretizedRandomGPFunction
from semi_synthetic_dataset_MimicIV import SyntheticOutcomeGenerator, SyntheticTreatment


num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def tensor_to_pddf0(tensor,tensor_static=None, val_names=None, time_names=None, id_names=None,val_names_static=None,data_name=None):

    if val_names == None:
        val_names = [f"val{i}" for i in range(tensor.size(2))]

    if time_names == None:
        
        time_names = [f"time{i}" for i in range(tensor.size(0))]

    if id_names == None:
        if data_name != None:
            id_names = [f"id{i}"+data_name for i in range(tensor.size(1))]
        else:
            id_names = [f"id{i}" for i in range(tensor.size(1))]
            
            
    data_np = tensor.numpy()
    reshaped = data_np.transpose(1, 0, 2).reshape(-1, tensor.size(2))
    id_repeat = [id for id in id_names for _ in range(tensor.size(0))]
    time_tile = time_names * len(id_names)
    multi_index = pd.MultiIndex.from_arrays([id_repeat, time_tile], names=["subject_id", "time"])
    df_dynamic = pd.DataFrame(reshaped, columns=val_names, index=multi_index)
        
    if tensor_static !=None:
        if val_names_static == None:
            val_names_static = [f"valS{i}" for i in range(tensor_static.size(2))] 
        
        data_np_static = tensor_static.numpy()
        reshaped_static = data_np_static.transpose(1, 0, 2).reshape(-1, tensor_static.size(2))
        
        df_static = pd.DataFrame(reshaped_static, columns=val_names_static, index=pd.Index(id_names, name="subject_id"))
        return df_dynamic, df_static 
    else:
        return df_dynamic
    
    
def tensor_to_pddf(tensor,tensor_static=None, val_names=None, time_names=None, id_names=None,val_names_static=None,data_name=None):

    if val_names == None:
        val_names = [i for i in range(tensor.size(2))]

    if time_names == None:
        
        time_names = [i for i in range(tensor.size(0))]

    if id_names == None:
        if data_name != None:
            id_names = [i+data_name for i in range(tensor.size(1))]
        else:
            id_names = [i for i in range(tensor.size(1))]
            
    data_np = tensor.numpy()
    reshaped = data_np.transpose(1, 0, 2).reshape(-1, tensor.size(2))
    id_repeat = [id for id in id_names for _ in range(tensor.size(0))]
    time_tile = time_names * len(id_names)
    multi_index = pd.MultiIndex.from_arrays([id_repeat, time_tile], names=["subject_id", "time"])
    df_dynamic = pd.DataFrame(reshaped, columns=val_names, index=multi_index)
        
    if tensor_static !=None:
        if val_names_static == None:
            val_names_static = [f"valS{i}" for i in range(tensor_static.size(2))] 
        
        data_np_static = tensor_static.numpy()
        reshaped_static = data_np_static.transpose(1, 0, 2).reshape(-1, tensor_static.size(2))
        
        df_static = pd.DataFrame(reshaped_static, columns=val_names_static, index=pd.Index(id_names, name="subject_id"))
        return df_dynamic, df_static 
    else:
        return df_dynamic    

def pddf_to_tensor(df_dynamic, df_static=None):

    subject_ids = df_dynamic.index.get_level_values("subject_id").unique().tolist()#df_static.index.tolist()
    time_steps = df_dynamic.index.get_level_values("time").unique().tolist()
    
    subject_count = len(subject_ids)
    time_count = len(time_steps)
    dynamic_features = df_dynamic.shape[1]
    

    id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    time_to_idx = {t: i for i, t in enumerate(time_steps)}
    tensor = torch.zeros(time_count, subject_count, dynamic_features)

    for (sid, time), row in df_dynamic.iterrows():
        i = id_to_idx[sid]
        t = time_to_idx[time]
        tensor[t, i, :] = torch.tensor(row.values, dtype=torch.float32)

    if df_static != None:
        static_features = df_static.shape[1]
        tensor_static = torch.zeros(1, subject_count, static_features)
        for sid, row in df_static.iterrows():
            i = id_to_idx[sid]
            tensor_static[0, i, :] = torch.tensor(row.values, dtype=torch.float32)

        return tensor, tensor_static
    else:
        return tensor
    

path_data = '/Users/jaschob/Desktop/data_m4/save_mimiciv_csv/'
with open(path_data + "data_val_mimiciv2.pkl", "rb") as datei:
    data_validate = pickle.load(datei)  # t_validate, x_validate,y_validate

with open(path_data + "data_train_mimiciv2.pkl", "rb") as datei:
    data_train = pickle.load(datei)

with open(path_data + "data_test_mimiciv2.pkl", "rb") as datei:
    data_test = pickle.load(datei)

# Steuerung u
with open(path_data + "data_val_mimiciv_value_u2.pkl", "rb") as datei:
    u_validate = pickle.load(datei)

with open(path_data + "data_train_mimiciv_value_u2.pkl", "rb") as datei:
    u_train = pickle.load(datei)

with open(path_data + "data_test_mimiciv_value_u2.pkl", "rb") as datei:
    u_test = pickle.load(datei)

t_validate, x_validate, y_validate = data_validate
t_train, x_train, y_train = data_train
t_test, x_test, y_test = data_test



val_list_dynamic = [5, 9, 14, 15]+[4,6,7,8,10,11,12,13]+list(range(19, 30))
val_list_static = list(range(30, 41))

name_dynamic1 = [
    'SOFA',
    'creatinine',
    'bilirubin_total',
    'alt',
    'index',
    'aniongap',
    'bicarbonate',
    'bun',
    'dbp',
    'platelet',
    'rdw',
    'sbp',
    'SOFA_mask',
    'aniongap_mask',
    'bicarbonate_mask',
    'bun_mask',
    'creatinine_mask',
    'dbp_mask',
    'platelet_mask',
    'rdw_mask',
    'sbp_mask',
    'bilirubin_total_mask',
    'alt_mask']



name_static = ['admission_age',
               'height',
               'weight',
               'male',
               'admission_age_mask',
               'height_mask',
               'weight_mask',
               'male_mask',
               'Vancomycinstat',
               'Piperacillin-Tazobactamstat',
               'Ceftriaxonstat']


######
train_size, validate_size, test_size = x_train.size(), x_validate.size(), x_test.size()
x = torch.cat([x_train, x_validate,x_test],dim=1)
x_all_dynamic, x_all_static = tensor_to_pddf(x[..., val_list_dynamic],x[:1,:, val_list_static], val_names=name_dynamic1,val_names_static=name_static)
#####



generator1 = SyntheticOutcomeGenerator(
    exogeneous_vars=['creatinine','bilirubin_total','alt'],
    exog_dependency=RandomFourierFeaturesFunction(input_dim=3, gamma=0.005, scale=40.0),
    exog_weight=1.0,
    endo_dependency=DiscretizedRandomGPFunction(kernels=(Matern(length_scale=25., nu=2.5),WhiteKernel(noise_level= 0.005))),
    endo_rand_weight=0.5,
    endo_spline_weight=2.,
    outcome_name='y1'#Sofa
)


for i in range(x.size(1)):
    x_all_dynamic.iloc[x.size(0)*i:x.size(0)*(i+1)] = x_all_dynamic.iloc[x.size(0)*i:x.size(0)*(i+1)].ffill().bfill()
    
x_all_dynamics_with_y1 = generator1.simulate_untreated(x_all_dynamic, x_all_static)
x_all_dynamic = pd.concat([x_all_dynamic, x_all_dynamics_with_y1['y1']], axis=1)

x_syn = pddf_to_tensor(pd.concat([x_all_dynamics_with_y1['y1']], axis=1))
#x_syn = pddf_to_tensor(x_all_dynamics_with_y1['y1'])
x_train_syn = x_syn[:,:x_train.size(1),:]
x_validate_syn = x_syn[:,x_train.size(1):x_train.size(1)+x_validate.size(1),:]
x_test_syn = x_syn[:,x_train.size(1)+x_validate.size(1):,:]


name_u = ['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']
u_train[torch.isnan(u_train)] = 0
u_test[torch.isnan(u_test)] = 0
u_validate[torch.isnan(u_validate)] = 0

u_validate_dynamic    = tensor_to_pddf(u_validate, val_names=name_u)
u_test_dynamic        = tensor_to_pddf(u_test, val_names=name_u)
u_train_dynamic       = tensor_to_pddf(u_train, val_names=name_u)

u_all_dynamic = pd.concat([u_train_dynamic, u_validate_dynamic,u_test_dynamic], axis=0)

###
# Treatment 1: u1
treatment_generator1 = SyntheticTreatment(
    confounding_vars=['creatinine'],
    confounder_outcomes=["y1"],
    confounding_dependency=RandomFourierFeaturesFunction(input_dim=1, gamma=0.001, scale=5.0),  # scale kleiner
    window=3,
    conf_outcome_weight=0.3,  
    conf_vars_weight=0.3,     
    bias=-2.0,                
    full_effect=-1.0,
    effect_window=20,
    treatment_name="u1"
)

# Treatment 2: u2
treatment_generator2 = SyntheticTreatment(
    confounding_vars=['bilirubin_total','alt'],
    confounder_outcomes=["y1"],
    confounding_dependency=RandomFourierFeaturesFunction(input_dim=2, gamma=0.001, scale=5.0),  # scale kleiner
    window=3,
    conf_outcome_weight=0.3,  
    conf_vars_weight=0.1,     
    bias=-2.0,                
    full_effect=-1.0,
    effect_window=20,
    treatment_name="u2"
)


treatments = [treatment_generator1,treatment_generator2]

patient_df = x_all_dynamic.loc[0].rename_axis("hours_in")

t = 0 
outcome_name = "y1"

treatment_ranges = []
treated_outcomes = []
treat_flags = []

for generator in treatments:
    proba = generator.treatment_proba(patient_df, t)
    treat_flag = proba > 0.5
    treat_flags.append(treat_flag)

    trange, out = generator.get_treated_outcome(patient_df, t, outcome_name, treat_proba=proba, treat=treat_flag)
    treatment_ranges.append(trange)
    treated_outcomes.append(out)
    
    
treatments = [treatment_generator1,treatment_generator2]
synthetic_treatments = pd.DataFrame(index=patient_df.index)

for tr_generator in treatments:
    probs = []
    for t in patient_df.index:
        proba = tr_generator.treatment_proba(patient_df, t)
        probs.append(proba[0])
    synthetic_treatments[tr_generator.treatment_name] = probs

print(synthetic_treatments.head())


####
import pandas as pd

patient_df = x_all_dynamic.loc[0].rename_axis("hours_in")

treatments = [treatment_generator1, treatment_generator2]
outcome_name = "y1"
u_all_syn = torch.zeros(x.size(0),x.size(1),2)
x_all_syn_treat = torch.zeros(x.size(0), x.size(1),2)



#train, val, test


for pation in range(len(x_all_dynamic.index.get_level_values("subject_id").unique())):    
    patient_df = x_all_dynamic.loc[pation].rename_axis("hours_in")
    synthetic_treatments = pd.DataFrame(index=patient_df.index)
    synthetic_outcomes = pd.DataFrame(index=patient_df.index)
    print(pation)
    for tr_generator in treatments:
        probs = []
        treated_outcome = []
        
        for t in patient_df.index:
            
            proba = tr_generator.treatment_proba(patient_df, t)
            probs.append(proba[0])
            
            
            treat_flag = proba > 0.5
            
            trange, out = tr_generator.get_treated_outcome(
                patient_df,
                t,
                outcome_name,
                treat_proba=proba,
                treat=treat_flag
            )
    
            treated_outcome.append(out.loc[t] if t in out.index else patient_df.loc[t, outcome_name])
        
    
        synthetic_treatments[tr_generator.treatment_name] = probs
        synthetic_outcomes[tr_generator.treatment_name + "_outcome"] = treated_outcome

    u_all_syn[:, pation, :] = torch.from_numpy(synthetic_treatments[['u1', 'u2']].to_numpy())
    x_all_syn_treat[:, pation, :] = torch.from_numpy(synthetic_outcomes[['u1_outcome', 'u2_outcome']].to_numpy())
          
x_train_syn_treat =x_all_syn_treat[:,:x_train.size(1),:]
x_validate_syn_treat =x_all_syn_treat[:,x_train.size(1):x_train.size(1)+x_validate.size(1),:]
x_test_syn_treat =x_all_syn_treat[:,x_train.size(1)+x_validate.size(1):,:]    

u_train_syn =u_all_syn[:,:x_train.size(1),:]
u_validate_syn =u_all_syn[:,x_train.size(1):x_train.size(1)+x_validate.size(1),:]
u_test_syn =u_all_syn[:,x_train.size(1)+x_validate.size(1):,:] 


x_train_syn = x_syn[:,:x_train.size(1),:]
x_validate_syn = x_syn[:,x_train.size(1):x_train.size(1)+x_validate.size(1),:]
x_test_syn = x_syn[:,x_train.size(1)+x_validate.size(1):,:]



path_data = '/Users/jaschob/Desktop/semi_syn_mimic4/'

with open(path_data+"x_train_syn_treat.pkl", "wb") as datei:
    pickle.dump(x_train_syn_treat, datei)
with open(path_data+"x_validate_syn_treat.pkl", "wb") as datei:
    pickle.dump(x_validate_syn_treat, datei)    
with open(path_data+"x_test_syn_treat.pkl", "wb") as datei:
    pickle.dump(x_test_syn_treat, datei)
    
with open(path_data+"u_train_syn.pkl", "wb") as datei:
    pickle.dump(u_train_syn, datei)
with open(path_data+"u_validate_syn.pkl", "wb") as datei:
    pickle.dump(u_validate_syn, datei)
with open(path_data+"u_test_syn.pkl", "wb") as datei:
    pickle.dump(u_test_syn, datei)
    
    
with open(path_data+"x_train_syn.pkl", "wb") as datei:
    pickle.dump(x_train_syn, datei)
with open(path_data+"x_validate_syn.pkl", "wb") as datei:
    pickle.dump(x_validate_syn, datei)        
with open(path_data+"x_test_syn.pkl", "wb") as datei:
    pickle.dump(x_test_syn, datei)    


with open(path_data+"t_train.pkl", "wb") as datei:
    pickle.dump(t_train[:,:,:2], datei)
with open(path_data+"t_validate.pkl", "wb") as datei:
    pickle.dump(t_validate[:,:,:2], datei)        
with open(path_data+"t_test.pkl", "wb") as datei:
    pickle.dump(t_test[:,:,:2], datei)    

with open(path_data+"x_all_dynamics_with_y1.pkl", "wb") as datei:
    pickle.dump( x_all_dynamics_with_y1, datei)  

