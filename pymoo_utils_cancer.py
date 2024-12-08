from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Choice, Real, Binary
import numpy as np
from src.utils.data_utils import process_data
import utils_paper_cancer as utils
import itertools

from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from optuna.samplers import TPESampler, GridSampler
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from src.utils.cancer_simulation import get_confounding_params
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text


class OnestepPymooProblem(ElementwiseProblem):
    # Function to perform one step treatment optimization with DoseAI using Pymoo
    def __init__(self,treat1_list=[0,3,5,7], treat2_list=[0,1.5,1.75,2], delta_tox=5, offset=0, model=None, data=None,treat_pre=None, real=False,scaling_data=None,max_time=None,seed=None, discrete_treatment=True, **kwargs):
        
        # inputs: 
        # treat1_list (list): List of potential Categorical Treatment dosages (here chemotherapy)
        # treat2_list (list): List of potential Categorical Treatment dosages (here radiotherapy)
        # delta_tox (int): Maximum allowed weight losses, default: 5
        # offset: Observation_horizon describing the time span for which the patient is observed before the forecast is made, default: 0
        # model (Neural CDE): Encoder of DoseAI
        # data (dict): Dict of synthetic cancer data including mean and std.
        # treat_pre (tensor): Tensor representing the past treatments administered to the patient, default: None
        # real (boolean): Flag indicating, whether to use real data or synthetic data
        # scaling_data (dict): Dictionary including the mean and std. of the variables, default: None
        # max_time: Forecast_horizon describing the time span for which treatment optimization is performed
        # seed (int): Seed to simulate the distributional parameters for patient simulation, default: None
        # discrete_treatment (boolean): Variable indicating, whether dose optimization should focus on discrete/categorical dosages or continuous/real-valued dosages, default: True
        
        if discrete_treatment:
            vars = {
                "treat1": Choice(options=treat1_list),
                "treat2": Choice(options=treat2_list),
            }
        else:
            vars = {
                "treat1": Real(bounds=(treat1_list[0],treat1_list[-1])),
                "treat2": Real(bounds=(treat2_list[0],treat2_list[-1])),
                "treat1_index": Binary(),
                "treat2_index": Binary()
            }
        
        self.discrete_treatment=discrete_treatment
        self.delta_tox=delta_tox
        self.ind=0
        self.offset=offset
        self.treat=np.zeros(shape=[1,self.offset+1,2])
        self.treat[:,:offset,:]=treat_pre
        self.model=model
        self.data=data
        self.real=real
        self.scaling_data=scaling_data
        self.max_time=max_time
        self.seed=seed
        
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=1, **kwargs)
        
    def _evaluate(self, X, out, *args, **kwargs):
        #delta_tox describes the maximum accepted side effects
        if not self.discrete_treatment:
            if X["treat1_index"]:
                 self.treat[0,self.offset,0] = X["treat1"]
            else:
                self.treat[0,self.offset,0] = 0
            if X["treat2_index"]:
                 self.treat[0,self.offset,1] = X["treat2"]
            else:
                self.treat[0,self.offset,1] = 0
        else:
            self.treat[0,self.offset,0] = X["treat1"]
            self.treat[0,self.offset,1] = X["treat2"]
        
        # Unscaled toxicity variable as initial value
        start_toxic = self.data["toxicity"][self.ind:self.ind+1,0]
        #start_toxic = (start_toxic*self.data["output_toxicity_stds"])+self.data["output_toxicity_means"]
        
        # Indicating whether using DoseAI or synthetic data generation model
        if not self.real:
            pred_neuralcde = utils.predict_opt(self.model, self.data, a_loss='spearman',offset=self.offset, a=self.treat)
            pred_neuralcde[:,:,1]=(pred_neuralcde[:,:,1]*self.data["output_toxicity_stds"])+self.data["output_toxicity_means"]
            pred_neuralcde[:,:,0]=(pred_neuralcde[:,:,0]*self.data["output_stds"])+self.data["output_means"]
            
            # F is target
            out["F"] = pred_neuralcde[:,-1,0].cpu().detach().numpy()
            # G are side conditions
            out["G"] = start_toxic-pred_neuralcde[:,-1,1].cpu().detach().numpy()-self.delta_tox
        else:
            chemo=X["treat1"]
            radio=X["treat2"]
            
            simulated_real_data_, processed_simulated_real_data_ = simulate_true_model(self.offset, output_earlier=copy.deepcopy(self.data), initial=False, scaling_data=self.scaling_data, max_time=self.max_time, chemo=chemo,radio=radio, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=self.seed)
            
            # F is target
            out["F"] = processed_simulated_real_data_["unscaled_outputs"][:,self.offset,0]
            
            # G are side conditions
            out["G"] = start_toxic-processed_simulated_real_data_["unscaled_outputs_toxicity"][:,self.offset,0]-self.delta_tox
        
class MultistepPymooProblem(ElementwiseProblem):
    # Wrapper to perform treatment optimization with DoseAI using pymoo
    def __init__(self,treat1_list=[0,3,5,7], treat2_list=[0,1.5,1.75,2], delta_tox=5, offset=0, model=None, model_decoder=None, data=None,treat_pre=None, real=False,scaling_data=None,max_time=None,seed=None, discrete_treatment=True, **kwargs):
        
        # inputs: 
        # treat1_list (list): List of potential Categorical Treatment dosages (here chemotherapy)
        # treat2_list (list): List of potential Categorical Treatment dosages (here radiotherapy)
        # delta_tox (int): Maximum allowed weight losses, default: 5
        # offset: Observation_horizon describing the time span for which the patient is observed before the forecast is made, default: 0
        # model (Neural CDE): Encoder of DoseAI
        # data (dict): Dict of synthetic cancer data including mean and std.
        # treat_pre (tensor): Tensor describing the past treatments, default: None
        # real (boolean): Indicating, whether to use real data or synthetic data
        # scaling_data (dict): Dictionary including the mean and std. of the variables, default: None
        # max_time: Forecast_horizon describing the time span for which treatment optimization is performed
        # seed (int): Seed to simulate the distributional parameters for patient simulation, default: None
        # discrete_treatment (boolean): Variable indicating, whether dose optimization should focus on discrete/categorical dosages or continuous/real-valued dosages, default: True
        
        if discrete_treatment:
            for i in range(offset,max_time):
                if i==offset:
                    vars = {
                        str("treat1_"+str(i)): Choice(options=treat1_list),
                        str("treat2_"+str(i)): Choice(options=treat2_list),
                    }
                else:
                    vars[str("treat1_"+str(i))]=Choice(options=treat1_list)
                    vars[str("treat2_"+str(i))]=Choice(options=treat2_list)
        
        else:
            for i in range(offset,max_time):
                if i==offset:
                    vars = {
                        str("treat1_"+str(i)): Real(bounds=(treat1_list[0],treat1_list[-1])),
                        str("treat2_"+str(i)): Real(bounds=(treat2_list[0],treat2_list[-1])),
                        str("treat1_index_"+str(i)): Binary(),
                        str("treat2_index_"+str(i)): Binary()
                    }
                else:
                    vars[str("treat1_"+str(i))]= Real(bounds=(treat1_list[0],treat1_list[-1]))
                    vars[str("treat2_"+str(i))]= Real(bounds=(treat2_list[0],treat2_list[-1]))
                    vars[str("treat1_index_"+str(i))] = Binary()
                    vars[str("treat2_index_"+str(i))] = Binary()
        
        self.discrete_treatment=discrete_treatment
        self.delta_tox=delta_tox
        self.ind=0
        self.offset=offset
        self.treat=np.zeros(shape=[1,max_time,2])
        self.treat[:,:offset,:]=treat_pre
        self.model=model
        self.model_decoder=model_decoder
        self.data=data
        self.real=real
        self.scaling_data=scaling_data
        self.max_time=max_time
        self.seed=seed

        super().__init__(vars=vars, n_obj=1, n_ieq_constr=1, **kwargs)
        
    def _evaluate(self, X, out, *args, **kwargs):
        #delta_tox describes the maximum accepted side effects
        
        for i in range(self.offset,self.max_time):
            self.treat[0,i,0] = X[str("treat1_"+str(i))]
            self.treat[0,i,1] = X[str("treat2_"+str(i))]
        
            if not self.discrete_treatment:
                if X["treat1_index_"+str(i)]:
                     self.treat[0,i,0] = X["treat1_"+str(i)]
                else:
                    self.treat[0,i,0] = 0
                if X["treat2_index_"+str(i)]:
                     self.treat[0,i,1] = X["treat2_"+str(i)]
                else:
                    self.treat[0,i,1] = 0
            else:
                self.treat[0,i,0] = X[str("treat1_"+str(i))]
                self.treat[0,i,1] = X[str("treat2_"+str(i))]
        
        #usage of unscaled toxicity variable as initial value
        start_toxic = self.data["toxicity"][self.ind:self.ind+1,0]

        if not self.real:
            pred_neuralcde = utils.predict_decoder_opt(self.model, self.model_decoder, self.data, offset=self.offset, max_horizon=self.max_time, a=self.treat)
            pred_neuralcde[:,:,1]=(pred_neuralcde[:,:,1]*self.data["output_toxicity_stds"])+self.data["output_toxicity_means"]
            pred_neuralcde[:,:,0]=(pred_neuralcde[:,:,0]*self.data["output_stds"])+self.data["output_means"]
            # F is aim
            out["F"] = pred_neuralcde[:,-1,0].cpu().detach().numpy()
            # G are side conditions
            out["G"] = start_toxic-pred_neuralcde[:,-1,1].cpu().detach().numpy()-self.delta_tox
        else:
            simulated_real_data_=copy.deepcopy(self.data)
            for i in range(self.offset,self.max_time):
                chemo=X["treat1_"+str(i)]
                radio=X["treat2_"+str(i)]
                simulated_real_data_, processed_simulated_real_data_ = simulate_true_model(i, output_earlier=simulated_real_data_, initial=False, scaling_data=self.scaling_data, max_time=self.max_time, chemo=chemo,radio=radio, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=self.seed)         
                
            # F is aim
            out["F"] = processed_simulated_real_data_["unscaled_outputs"][:,-1,0]
            
            # G are side-conditions
            out["G"] = start_toxic-processed_simulated_real_data_["unscaled_outputs_toxicity"][:,-1,0]-self.delta_tox


def MultistepPymoo(treat1_list=[0,3,5,7], treat2_list=[0,1,2,3],delta_tox=6,seed=1,offset=0, model=None, model_decoder=None, max_time=5, scaling_data=None, real=False, alg='optuna', n_gen=5, pop_size=5,discrete_treatment=True,termination='ftol'):
    # Function to perform treatment optimization with DoseAI or synthetic data generation model using Pymoo
    
    # inputs: 
    # treat1_list (list): List of potential Categorical Treatment dosages (here chemotherapy)
    # treat2_list (list): List of potential Categorical Treatment dosages (here radiotherapy)
    # delta_tox (int): Maximum allowed weight losses
    # seed (int): Seed to simulate the distributional parameters for patient simulation, default: 1
    # offset: Observation_horizon describing the time span for which the patient is observed before the forecast is made, default: 0
    # model (Neural CDE): Encoder of DoseAI
    # model_decoder (Neural CDE): Decoder of DoseAI
    # max_time: Forecast_horizon describing the time span for which treatment optimization is performed
    # scaling_data (dict): Dictionary including the mean and std. of the variables, default: None
    # real (boolean): Indicating, whether DoseAI's predictions are used or synthetic data generation model
    # alg (str): Variable indicating the optimization algorithm, ['optuna','ga','grid'], default: 'optuna'
    # n_gen (int): Termination specific parameters. For 'ftol' describe the number of skipped iteration for evaluating the termination, else: number of generations, default: 5
    # pop_size (int): the number of individuals in generation, default: 5
    # discrete_treatment (boolean): Variable indicating, whether dose optimization should focus on discrete/categorical dosages or continuous/real-valued dosages, default: True
    # termination (string): Indicating, which termination criterion should be used, default: 'ftol'
    
    # outputs:
    # final_treat (tensor): Including the final treatments
    # simulated_real_data (dict): Dictionary with the synthetic disease predictions for the optimal treatment
    # processed_simualted_real_data (dict): Dictionary with preprocessed synthetic disease predictions for the optimal treatment
    # pred_neuralcde (tensor): Predictions with DoseAI for the optimal treatment

    
    # Simulating data with the underlying data generation model    
    simulated_real_data, processed_simulated_real_data = simulate_true_model(0, output_earlier=None, initial=True, scaling_data=scaling_data, max_time=max_time, chemo=0,radio=0, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=seed)
    
    if alg=='optuna':
        algorithm=Optuna(sampler=TPESampler())
    elif alg=='grid':
        search_space={}
        for i in range(offset,max_time):
            search_space["treat1_"+str(i)] = treat1_list
            search_space["treat2_"+str(i)] = treat2_list
        algorithm=Optuna(sampler=GridSampler(search_space))
    elif alg=='ga':
        algorithm = MixedVariableGA(pop_size=pop_size, survival=FitnessSurvival())
    
    if not real:
        # Tensor with (historical) treatments
        treat_pre = np.zeros(shape=[1,max_time,2])
        for i in range(offset,max_time):
            
            if i ==max_time-1:
                optimization_problem = OnestepPymooProblem(treat1_list=treat1_list, treat2_list=treat2_list, delta_tox=delta_tox,offset=i,treat_pre=treat_pre[:,:i,:],data=processed_simulated_real_data,model=model,discrete_treatment=discrete_treatment)
            elif i == offset:
                optimization_problem = MultistepPymooProblem(treat1_list=treat1_list, treat2_list=treat2_list,delta_tox=delta_tox,offset=offset, model=model, model_decoder=model_decoder, data=processed_simulated_real_data, max_time=max_time,discrete_treatment=discrete_treatment)
            else:
                optimization_problem = MultistepPymooProblem(treat1_list=treat1_list, treat2_list=treat2_list, delta_tox=delta_tox,offset=i,treat_pre=treat_pre[:,:i,:],model=model,model_decoder=model_decoder,data=processed_simulated_real_data,max_time=max_time,discrete_treatment=discrete_treatment)
            optimization_problem.data=copy.deepcopy(processed_simulated_real_data)
            
            if termination=='ftol':
                termination_crit = RobustTermination(SingleObjectiveSpaceTermination(only_feas=False,tol=0.05,n_skip=n_gen))
            else:
                termination_crit = ("n_gen",n_gen)
            
            # Optimization problem
            res = minimize(optimization_problem,
                           algorithm,
                           termination=termination_crit,
                           seed=1,
                           verbose=False)
            
            # Setting treatment to zero, if optimization can not find an optimal treatment
            if res.X is None:
                simulated_real_data, processed_simulated_real_data = simulate_true_model(i, output_earlier=simulated_real_data, initial=False, scaling_data=scaling_data, max_time=max_time, chemo=0, radio=0, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=seed)
                treat_pre[0,i,:] = 0
            
            else:
                simulated_real_data, processed_simulated_real_data = simulate_true_model(i, output_earlier=simulated_real_data, initial=False, scaling_data=scaling_data, max_time=max_time, chemo=list(res.X.values())[0], radio=list(res.X.values())[1], num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=seed)
            
                if not discrete_treatment:
                    if list(res.X.values())[2]:
                        treat_pre[0,i,0] = list(res.X.values())[0]
                    else:
                        treat_pre[0,i,0] = 0
                    if list(res.X.values())[3]:
                        treat_pre[0,i,1] = list(res.X.values())[1]
                    else:
                        treat_pre[0,i,1] = 0
                else:
                    treat_pre[0,i,:] = list(res.X.values())[:2]
            
            # Earlier stopping when cancer volume is zero
            if simulated_real_data["cancer_volume"][0,-1]==0:
                final_treat=treat_pre[0,:,:]
                processed_simulated_real_data = process_data(simulated_real_data.copy(),toxicity=True,continuous=True,scaling_data=scaling_data,treatment_testdata=True)
                
                pred_neuralcde = utils.predict_decoder_opt(model, model_decoder, processed_simulated_real_data, offset=offset, max_horizon=max_time, a=treat_pre)
                pred_neuralcde[:,:,1]=(pred_neuralcde[:,:,1]*processed_simulated_real_data["output_toxicity_stds"])+processed_simulated_real_data["output_toxicity_means"]
                pred_neuralcde[:,:,0]=(pred_neuralcde[:,:,0]*processed_simulated_real_data["output_stds"])+processed_simulated_real_data["output_means"]
                return final_treat, simulated_real_data, processed_simulated_real_data, pred_neuralcde
            
        final_treat=treat_pre[0,:,:]
        processed_simulated_real_data = process_data(simulated_real_data.copy(),toxicity=True,continuous=True,scaling_data=scaling_data,treatment_testdata=True)
        
        pred_neuralcde = utils.predict_decoder_opt(model, model_decoder, processed_simulated_real_data, offset=offset, max_horizon=max_time, a=treat_pre)
        pred_neuralcde[:,:,1]=(pred_neuralcde[:,:,1]*processed_simulated_real_data["output_toxicity_stds"])+processed_simulated_real_data["output_toxicity_means"]
        pred_neuralcde[:,:,0]=(pred_neuralcde[:,:,0]*processed_simulated_real_data["output_stds"])+processed_simulated_real_data["output_means"]
    else:
        optimization_problem = MultistepPymooProblem(treat1_list=treat1_list, treat2_list=treat2_list,delta_tox=delta_tox,offset=offset, model=model, model_decoder=model_decoder, data=processed_simulated_real_data,real=True,scaling_data=scaling_data,max_time=max_time,seed=seed,discrete_treatment=discrete_treatment)
        optimization_problem.data=copy.deepcopy(simulated_real_data)
        
        if termination=='ftol':
            termination_crit = RobustTermination(SingleObjectiveSpaceTermination(only_feas=False,tol=0.05,n_skip=n_gen))
        else:
            termination_crit = ("n_gen",n_gen)
        
        res = minimize(optimization_problem,
                       algorithm,
                       termination=termination_crit,
                       seed=1,
                       verbose=False)
        
        final_treat = np.zeros(shape=[1,max_time,2])
        if res.X is None:
            return None, None, None, None
        
        else:
            for i in range(offset,max_time):
                simulated_real_data, processed_simulated_real_data = simulate_true_model(i, output_earlier=simulated_real_data, initial=False, scaling_data=scaling_data, max_time=max_time, chemo=res.X["treat1_"+str(i)], radio=res.X["treat2_"+str(i)], num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=seed)
                if not discrete_treatment:
                    if res.X["treat1_index_"+str(i)]:    
                        final_treat[0,i,0]=res.X["treat1_"+str(i)]
                    else:
                        final_treat[0,i,0] = 0
                    if res.X["treat2_index_"+str(i)]:    
                        final_treat[0,i,1]=res.X["treat2_"+str(i)]
                    else:
                        final_treat[0,i,1] = 0
                else:
                    final_treat[0,i,0] = 0
                    final_treat[0,i,1] = 0
                if simulated_real_data["cancer_volume"][0,-1]==0:
                    pred_neuralcde=None    
                    return final_treat, simulated_real_data, processed_simulated_real_data, pred_neuralcde
                
        pred_neuralcde=None
        
    return final_treat, simulated_real_data, processed_simulated_real_data, pred_neuralcde

def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0

def simulate_true_model(time, output_earlier=None, initial=False, scaling_data=None, max_time=10, chemo=0,radio=0, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200):
    # Function to simulate the disease progression for DoseAI's optimal treatment
    
    # time (int): Maximum horizon for data generation
    # output_earlier (dict): Dictionary including the synthetic data of earlier timepoints, default: None
    # initial (boolean): true initializes the first iteration, default: False
    # scaling_data (dict): Dictionary including the mean and std. of the variables, default: None
    # max_time: Forecast_horizon describing the time span for which treatment optimization is performed
    # chemo (int): Chemotherapy dose
    # radio (int): radiotherapy dose
    # num_patients(int): Number of generated patients, default: 1
    # chemo_coeff (int): Bias on action policy for chemotherapy dosages
    # radio_coeff (int): Bias on action policy radiotherapy dosages
    # window_size (int): Controls the lookback of the treatment assignment policy
    # seed (int): Seed for generation of parameters
    
    np.random.seed(seed)
     
    # No earlier data
    if initial:
        simulation_params = get_confounding_params(
            int(num_patients),
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            toxicity=True
        )
        simulation_params["window_size"] = window_size  
        noise_terms_toxic = 0.0015 * np.random.randn(
            num_patients,
            max_time,
        )
    
        noise_terms = 0.01 * np.random.randn(
            num_patients,
            max_time,
        )
        simulation_params["noise_terms"]=noise_terms
        simulation_params["noise_terms_toxic"]=noise_terms_toxic
    else:
        simulation_params=output_earlier["simulation_params"]
        noise_terms=simulation_params["noise_terms"]
        noise_terms_toxic=simulation_params["noise_terms_toxic"]

    total_num_chemo_treatments = 1

    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params["initial_stages"]
    initial_volumes = simulation_params["initial_volumes"]
    initial_toxicity = simulation_params["initial_toxicity"]
    
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    kgs = simulation_params["kg"]
    kl1s = simulation_params["kl1"]
    kl2s = simulation_params["kl2"]
    kl3s = simulation_params["kl3"]
    patient_types = simulation_params["patient_types"]
    
    if initial:
        cancer_volume = np.zeros((num_patients, max_time+1))
        cancer_volume[:] = np.nan
        chemo_dosage = np.zeros((num_patients, max_time+1))
        chemo_dosage[:] = np.nan
        radio_dosage = np.zeros((num_patients, max_time+1))
        radio_dosage[:] = np.nan
        chemo_application_point = np.zeros((num_patients, max_time+1))
        chemo_application_point[:] = np.nan
        radio_application_point = np.zeros((num_patients, max_time+1))
        radio_application_point[:] = np.nan
        sequence_lengths = np.zeros((num_patients))
        toxic = np.zeros((num_patients, max_time+1))
        toxic[:] = np.nan
    else:
        cancer_volume = output_earlier["cancer_volume"]
        chemo_dosage = output_earlier["chemo_dosage"]
        radio_dosage = output_earlier["radio_dosage"]
        chemo_application_point = output_earlier["chemo_application"]
        radio_application_point = output_earlier["radio_application"]
        sequence_lengths = output_earlier["sequence_lengths"]
        toxic = output_earlier["toxicity"]
    
    for i in range(num_patients):
        noise = noise_terms[i]        
        
        # initial values
        if initial:
            cancer_volume[i, 0] = initial_volumes[i]
            toxic[i, 0] = initial_toxicity[i]
            sequence_lengths[i] = max_time

        else:
            alpha = alphas[i]
            beta = betas[i]
            beta_c = beta_cs[i]
            rho = rhos[i]
            K = Ks[i]
            noise_toxic = noise_terms_toxic[i]
            kg=kgs[i]
            kl1=kl1s[i]
            kl2=kl2s[i]
            kl3=kl3s[i]
            
            previous_chemo_dose = 0.0 if time == 0 else chemo_dosage[i, time - 1]
            
            # Action application
            radio_application_point[i, time] = radio
            radio_dosage[i, time] = radio

            current_chemo_dose = chemo
            chemo_application_point[i, time] = chemo
            # Update chemo dosage
            chemo_dosage[i, time] = (
                previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
                + current_chemo_dose
            )
            
            if cancer_volume[i, time]>0:
    
                cancer_volume[i, time + 1] = cancer_volume[i, time] * (
                    1
                    + rho * np.log(K / cancer_volume[i, time])
                    - beta_c * chemo_dosage[i, time]
                    - (alpha * radio_dosage[i, time] + beta * radio_dosage[i, time] ** 2)
                    + noise[time]
                )
                
                toxic[i, time + 1] = toxic[i, time] * (
                    1
                    + kg * toxic[i, time] *(1-toxic[i,time]/(initial_toxicity[i])) 
                    - kl1 * chemo_dosage[i, time]
                    - kl2 * radio_dosage[i, time]
                    - kl3 * cancer_volume[i, time]
                    + noise_toxic[time]
                )
            else:
                cancer_volume[i, time + 1] = 0
                toxic[i, time + 1] = 0
                
            tumour_death_threshold=calc_volume(13)
            
            if cancer_volume[i, time + 1] > tumour_death_threshold:
                cancer_volume[i, time + 1] = tumour_death_threshold
                sequence_lengths[i] = int(time+1)
                break  # patient death
            elif cancer_volume[i, time + 1] <= 0:
                cancer_volume[i, time + 1] = 0
                sequence_lengths[i] = int(time+1)
                #break  # patient death
            else:
                sequence_lengths[i] = max_time
                
                
            
    simulated_real_data = {
        "cancer_volume": cancer_volume,
        "chemo_dosage": chemo_dosage,
        "radio_dosage": radio_dosage,
        "chemo_application": chemo_application_point,
        "radio_application": radio_application_point,
        "sequence_lengths": sequence_lengths,
        "toxicity": toxic,
        "patient_types": patient_types,
        "simulation_params": simulation_params
    }
    
    processed_simulated_real_data = process_data(simulated_real_data.copy(),toxicity=True,continuous=True,scaling_data=scaling_data,treatment_testdata=True)
    
    return simulated_real_data, processed_simulated_real_data

def plot_individual_patients(model, model_decoder, treat1_list=[0,3,5,7], treat2_list=[0,1,2,3], delta_list = [5,7,9,12,15,18,20], seed=0, offset=0, max_time=5, scaling_params=None, alg='optuna', n_gen=100, plot_size=40, plot=True, plt_save=False, path=None, sim=True, CDE_opt_trt_in_real_model=None, Real_opt_trt_progression=None, opt_treatment_multi=None, real_opt_treatment_multi=None, legend=True, ylims=None, discrete_treatment=True,termination='ftol',pop_size=100,real_sim=True):

    # function to simulate and plot DoseAI's optimal treatments
    
    # inputs: 
    # model (Neural CDE): Encoder of DoseAI
    # model_decoder (Neural CDE): Decoder of DoseAI
    # treat1_list (list): List of potential Categorical Treatment dosages (here chemotherapy)
    # treat2_list (list): List of potential Categorical Treatment dosages (here radiotherapy)
    # delta_list (list): List of maximum allowed weight losses
    # seed (int): Seed to simulate the distributional parameters for patient simulation, default: 0
    # offset: Observation_horizon describing the time span for which the patient is observed before the forecast is made, default: 0
    # max_time: Forecast_horizon describing the time span for which treatment optimization is performed
    # scaling_params (dict): Dictionary including the mean and std. of the variables, default: None
    # alg (str): Variable indicating the optimization algorithm, ['optuna','ga','grid'], default: 'optuna'
    # n_gen (int): Termination specific parameters. For 'ftol' describe the number of skipped iteration for evaluating the termination, else: number of generations, default: 100
    # plot_size (int): Fontsize of the plot, default: 40
    # plot (boolean): Indicating, whether to plot the results or not, default: True
    # plt_save(boolean): Indicating, whether to save the plot or not, default: False
    # path: Path to save the plot, default: None
    # sim (boolean): Indicating, whether to simulate the results or not
        
    # CDE_opt_trt_in_real_model (tensor): Predictions of Simulated optimal treatment with DoseAI in synthetic data generation model, default: None
    # Real_opt_trt_progression (tensor): Predictions of simulated optimal treatment with synthetic data generation model, default: None
    # opt_treatment_multi (dict): Dictionary with optimal treatments simulated with DoseAI for different maximum allowed weight losses, default: None
    # real_opt_treatment_multi (dict): Dictionary with optimal treatments simulated with synthetic data generation model for different maximum allowed weight losses, default: None
  
    # legend (boolean): Indicating, whether a legend is plotted or not, default: True
    # ylims ([int ymin, int ymax]): limits for the y-axis
    # discrete_treatment (boolean): Variable indicating, whether dose optimization should focus on discrete/categorical dosages or continuous/real-valued dosages, default: True
    # termination (string): Indicating, which termination criterion should be used, default: 'ftol'
    # pop_size (int): the number of individuals in generation, default: 100
    # real_sim (boolean): Indicating, whether results with underlying known ODE should be generated or not
    
    # outputs:
    
    # CDE_opt_trt_in_real_model (tensor): Predictions of Simulated optimal treatment with DoseAI in synthetic data generation model
    # Real_opt_trt_progression (tensor): Predictions of simulated optimal treatment with synthetic data generation model
    # opt_treatment_multi (dict): Dictionary with optimal treatments simulated with DoseAI for different maximum allowed weight losses
    # real_opt_treatment_multi (dict): Dictionary with optimal treatments simulated with synthetic data generation model for different maximum allowed weight losses


    plt.rc('font', size=plot_size)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=plot_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=31.5)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=31.5)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    plt.rcParams['lines.markersize'] = 50

    fig,ax = plt.subplots(figsize=(14,14),dpi=400)
    ax.set_xlim(delta_list[0]-1,delta_list[-1]+1)
    ax.set_xticks(delta_list)
    ax.set_xticklabels(delta_list)
    ax.set_xlabel("Max allowed weight loss $\delta^{{tox}}$ in $kg$")
    ax.set_ylabel("Tumor Volume in $cm^3$")
    
    if sim:
        CDE_opt_trt_in_real_model = np.empty(shape=[len(delta_list),3])
        CDE_opt_trt_in_real_model[:] = np.nan
        CDE_opt_trt_in_real_model[:,0]=delta_list
        Real_opt_trt_progression = np.empty(shape=[len(delta_list),3])
        Real_opt_trt_progression[:] = np.nan
        Real_opt_trt_progression[:,0]=delta_list
    CDE_opt_trt = {}
    real_opt_trt = {}
    CDE_opt_trt_in_real_model_complete = {}
    Real_opt_trt_progression_complete = {}
    
    for i in range(len(delta_list)):
        if sim:
            # Simulating optimal treatment with Neural CDE model
            opt_treatment_multi, opt_treatment_in_sim_model_multi, processed_simulated_real_data_multi, predicted_data_multi = MultistepPymoo(treat1_list=treat1_list, treat2_list=treat2_list,delta_tox=delta_list[i],seed=seed,offset=offset, model=model, model_decoder=model_decoder, max_time=max_time, scaling_data=scaling_params, pop_size=pop_size, n_gen=n_gen, real=False, alg=alg,discrete_treatment=discrete_treatment)
            if opt_treatment_multi is not None:
                if (processed_simulated_real_data_multi["unscaled_outputs"]==0).any():
                    CDE_opt_trt_in_real_model[i,1] = 0
                    CDE_opt_trt_in_real_model[i,2] = 0
                else:
                    CDE_opt_trt_in_real_model[i,1] = processed_simulated_real_data_multi["unscaled_outputs"][0,-1,0]
                    CDE_opt_trt_in_real_model[i,2] = processed_simulated_real_data_multi["toxicity"][0,0]-processed_simulated_real_data_multi["unscaled_outputs_toxicity"][0,-1,0]
            
            # Simulating optimal treatment with real ODE model
            if real_sim:
                real_opt_treatment_multi, real_opt_treatment_in_sim_model_multi, real_processed_simulated_real_data_multi,_ = MultistepPymoo(treat1_list=treat1_list, treat2_list=treat2_list,delta_tox=delta_list[i],seed=seed,offset=offset, model=model, model_decoder=model_decoder, max_time=max_time, scaling_data=scaling_params, pop_size=pop_size, n_gen=n_gen, real=True, alg=alg,discrete_treatment=discrete_treatment)
            else:
                real_opt_treatment_multi=None
                real_opt_treatment_in_sim_model_multi=None
                real_processed_simulated_real_data_multi=None
            if real_opt_treatment_multi is not None:
                if (real_processed_simulated_real_data_multi["unscaled_outputs"]==0).any():
                    Real_opt_trt_progression[i,1] = 0
                    Real_opt_trt_progression[i,2] = 0
                else:
                    Real_opt_trt_progression[i,1] = real_processed_simulated_real_data_multi["unscaled_outputs"][0,-1,0]
                    Real_opt_trt_progression[i,2] = real_processed_simulated_real_data_multi["toxicity"][0,0]-real_processed_simulated_real_data_multi["unscaled_outputs_toxicity"][0,-1,0]
            CDE_opt_trt[delta_list[i]] = opt_treatment_multi
            real_opt_trt[delta_list[i]]=real_opt_treatment_multi
            CDE_opt_trt_in_real_model_complete[delta_list[i]] = processed_simulated_real_data_multi
            Real_opt_trt_progression_complete[delta_list[i]] = real_processed_simulated_real_data_multi
        if plot:
            # CDE Results
            if opt_treatment_multi is not None:
                # If toxicity too high, then plotting in red
                if (CDE_opt_trt_in_real_model[i,2])>delta_list[i]:
                    color='red'
                else:
                    color='aqua'
                    
                ax.scatter(delta_list[i], CDE_opt_trt_in_real_model[i,1],color=color)#, s=BIGGER_SIZE)
                if opt_treatment_multi is not None:
                    # Plotting value of toxicity
                    if (CDE_opt_trt_in_real_model[i,2])>delta_list[i]:
                        ax.annotate(str(np.round(CDE_opt_trt_in_real_model[i,2],decimals=1)), (delta_list[i],CDE_opt_trt_in_real_model[i,1]),size=24,va='center', ha='center')
            #Results for real model 
            if real_opt_treatment_multi is not None:
                ax.scatter(delta_list[i], Real_opt_trt_progression[i,1],color='black',marker='x')#, markersize=BIGGER_SIZE)
            
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='DoseAI satisfying $\delta^{{tox}}$', markerfacecolor='aqua', markersize=40),
                               Line2D([0], [0], marker='o', color='w', label='DoseAI violating $\delta^{{tox}}$', markerfacecolor='red', markersize=40),
                               Line2D([0], [0], marker='x', color='black', label='Data simulation model', markerfacecolor='black', markersize=40, linestyle=" ")]
            
            if ylims is not None:
                ax.set_ylim(ylims[0],ylims[1])
            if legend:
                ax.legend(handles=legend_elements)
    if plt_save:
        plt.savefig(path+'multistep'+str(seed)+'.jpg')
    
    return CDE_opt_trt_in_real_model, Real_opt_trt_progression, CDE_opt_trt, real_opt_trt, CDE_opt_trt_in_real_model_complete,Real_opt_trt_progression_complete

def plot_dynamic(volumes, chemo_doses, radio_doses,metric='cancer', plot_size=40, ylim=None,plt_save=False,path=None,legend=True):
    # function to plot the tumor volume or weight dynamic for DoseAI's optimal treatments
    # volumes (dict): Cancer volumes or weight dynamics for different maximum allowed weight losses
    # chemo_doses (dict): Optimal chemotherapy dosages for different maximum allowed weight losses
    # radio_doses (dict): Optimal radiotherapy dosages for different maximum allowed weight losses
    # metric (string): 'cancer' or 'weight' describing the metric to plot, default: 'cancer'
    # plot_size: Fontsize of the plot, default: 40
    # ylim ([int ymin, int ymax]): List incorporating the minimal and the maximal value of the y axis, default: None
    # plt_safe (boolean): Indicates, whether to safe the plot or not, default: False
    # path (boolean): Path for saving the plot, default: None

        
    # Define treatment cycles and delta^tox values
    #treatment_cycles = range(0, 6)  # 0 through 5, as there's no treatment at the last time point
    delta_tox_values = list(volumes.keys())
    
    chemo_doses2 = np.round(np.array(list(chemo_doses.values())),decimals=1)
    radio_doses2 = np.round(np.array(list(radio_doses.values())),decimals=1)
    
    new_doses = np.empty(chemo_doses2.shape, dtype=object)

    # Fill the new array with tuples
    for i in range(chemo_doses2.shape[0]):
        for j in range(chemo_doses2.shape[1]):
            new_doses[i, j] = (chemo_doses2[i, j], radio_doses2[i, j])
    
    # Create a figure and axis with quadratic aspect ratio
    plt.rc('font', size=plot_size)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=plot_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=31.5)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=31.5)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    plt.rcParams['lines.markersize'] = 50

    fig,ax = plt.subplots(figsize=(14,14),dpi=400)

    # Plot cancer volume for each delta^tox value
    for delta_tox in delta_tox_values[::-1]:
        line, = ax.plot(range(0, 7), volumes[delta_tox], label=fr'$\delta^{{tox}}={delta_tox}$')

    # Set labels
    ax.set_xlabel('Monthly Treatment Cycles')
    if metric=='cancer':
        ax.set_ylabel(r'Tumor Volume in $cm^3$')#, fontsize=label_size)  # LaTeX formatted label
        ax.set_title('Tumor Volume')
    elif metric=='weight':
        ax.set_ylabel(r'Weight in kg')#, fontsize=label_size)  # LaTeX formatted label
        ax.set_title('Body Weight')
    # Add legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1],labels[::-1])
    
    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major')
    
    if ylim is not None:
        ax.set_ylim(ylim)

    if plt_save:
        plt.savefig(path+metric+'.jpg')
        
def plot_dosages(chemo_doses=None,radio_doses=None, metric='chemo',plt_save=False,path=None,legend=False,location='best'):
    # metric: 'chemo','radio','twin'
    
    plot_size = 50
    plt.rc('font', size=plot_size)          # controls default text sizes
    plt.rc('axes', titlesize=50)           # fontsize of the axes title
    plt.rc('axes', labelsize=plot_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=41.5)        # fontsize of the tick labels
    plt.rc('ytick', labelsize=41.5)        # fontsize of the tick labels
    plt.rc('legend', fontsize=40)          # legend fontsize
    plt.rc('figure', titlesize=35)
    plt.rcParams['lines.markersize'] = 50  # Reduced marker size
    
    
    colors = ['tab:purple','tab:red','tab:green','tab:orange','tab:blue']
    #colors=colors[::-1]
    time = np.arange(0, 6, 1)
    
    markers = ['o','*','x','s','v']
    
    # Toxicity labels for legend
    toxicity_labels = [r'$\delta^{tox}=6$', r'$\delta^{tox}=9$', r'$\delta^{tox}=12$', r'$\delta^{tox}=15$', r'$\delta^{tox}=18$']
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 14), dpi=400)
    
    # Plot data for chemotherapy and radiotherapy with alpha=0.5
    for i in range(5):
        if metric=='chemo' or metric=='twin':
            ax1.scatter(time, chemo_doses[list(chemo_doses.keys())[i]], marker=markers[i], alpha=0.5,color=colors[i])
            ax1.step(np.append(time,time[-1]+1), np.append(chemo_doses[list(chemo_doses.keys())[i]],chemo_doses[list(chemo_doses.keys())[i]][-1]), where='post', alpha=0.5,color=colors[i])
        else:
            ax1.scatter(time, radio_doses[list(radio_doses.keys())[i]], marker=markers[i], alpha=0.5,color=colors[i])
            ax1.step(np.append(time,time[-1]+1), np.append(radio_doses[list(radio_doses.keys())[i]],radio_doses[list(radio_doses.keys())[i]][-1]), where='post', alpha=0.5,color=colors[i])
        #ax1.plot(time, chemo_doses[list(chemo_doses.keys())[i]], linestyle='-', marker='o', alpha=0.5,color=colors[i])
        
    # Customize first y-axis
    ax1.set_xlabel('Monthly Treatment Cycles')
    if metric == 'chemo' or metric=='twin':
        ax1.set_ylabel('Chemotherapy Dosage in $\\frac{mg}{m^{3}}$')
        plt.title('Chemotherapy')
    else:
        plt.title('Radiotherapy')
        ax1.set_ylabel('Radiotherapy Dosage in Gy')
    ax1.set_xlim([-0.3,6.3])
    ax1.tick_params(axis='y', labelcolor='black')
    
    if metric=='twin':
        # Create a second y-axis for radiotherapy data
        ax2 = ax1.twinx()
        for i in range(5):
            ax2.scatter(time, radio_doses[list(radio_doses.keys())[i]], marker='s', alpha=0.5,color=colors[i])
            ax2.step(time, radio_doses[list(radio_doses.keys())[i]], where='post', linestyle='--', alpha=0.5,color=colors[i])
            #ax2.plot(time, radio_doses[list(radio_doses.keys())[i]], linestyle='--', marker='s', alpha=0.5,color=colors[i])
            
            
        # Customize second y-axis
        ax2.set_ylabel('Radiotherapy Dosage in Gy')
        ax2.tick_params(axis='y', labelcolor='black')
        #ax2.set_ylim([-0.5,3.5])
        
        # Add dummy plots for chemotherapy and radiotherapy legend symbols
        chemotherapy_line, = ax1.plot([], [], color='black', marker='o', linestyle='-', label='Chemotherapy')
        radiotherapy_line, = ax2.plot([], [], color='black', marker='s', linestyle='--', label='Radiotherapy')
        
    if legend:
        legend_labels = toxicity_labels# + ['Chemo', 'Radio']
        legend_handles = [plt.Line2D([], [], color=colors[i],marker=markers[i], linestyle='-', alpha=0.5) for i in range(5)]# + [chemotherapy_line,radiotherapy_line]
        ax1.legend(handles=legend_handles, labels=legend_labels,loc=location)#, loc='upper left') #bbox_to_anchor=(0, 1.05), ncol=1)
    
    #plt.tight_layout()
    #plt.show()
    if plt_save:
        plt.savefig(path+metric+'.jpg')

        