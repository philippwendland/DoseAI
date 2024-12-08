# Adaptation from main.py from https://github.com/seedatnabeel/TE-CDE/blob/main/main.py
# Nabeel Seedat et al. “Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations”. In: Proceedings of the 39th International Conference on Machine Learning. Ed. by Kamalika Chaudhuri et al. Vol. 162.
# Proceedings of Machine Learning Research. PMLR, July 2022, pp. 19497–19521

"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018 &
https://github.com/ioanabica/Counterfactual-Recurrent-Network

Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
URL: https://www.nature.com/articles/s41598-017-13646-z

"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import (
    truncnorm, 
)
import itertools
from scipy.special import softmax
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Constants

# Spherical calculations - tumours assumed to be spherical per Winer-Muram et al 2002.
# URL: https://pubs.rsna.org/doi/10.1148/radiol.2233011026?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed
def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def calc_diameter(volume):
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0

# Tumour constants per
tumour_cell_density = 5.8 * 10.0**8.0  # cells per cm^3
tumour_death_threshold = calc_volume(13)  # assume spherical

tumour_size_distributions = {
    #"I": (1.72, 4.70, 0.3, 5.0),
    #"II": (1.96, 1.63, 0.3, 13.0),
    #"IIIA": (1.91, 9.40, 0.3, 13.0),
    #"IIIB": (2.76, 6.87, 0.3, 13.0),
    "IV": (7, 4.82, 4.5, 13.0),
}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {
    #"I": 1432,
    #"II": 128,
    #"IIIA": 1306,
    #"IIIB": 7248,
    "IV": 12840,
}



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions


def get_confounding_params(num_patients, chemo_coeff, radio_coeff, toxicity=False):
    """

    Get original simulation parameters, and add extra ones to control confounding

    :param num_patients:
    :param chemo_coeff: Bias on action policy for chemotherapy assignments
    :param radio_activation_group: Bias on action policy for chemotherapy assignments
    :return:
    """

    basic_params = get_standard_params(num_patients)
    patient_types = basic_params["patient_types"]
    tumour_stage_centres = [s for s in cancer_stage_observations if "IIIA" not in s]
    tumour_stage_centres.sort()

    d_max = calc_diameter(tumour_death_threshold)
    basic_params["chemo_sigmoid_intercepts"] = np.array(
        [d_max / 2.0 for i in patient_types],
    )
    basic_params["radio_sigmoid_intercepts"] = np.array(
        [d_max / 2.0 for i in patient_types],
    )

    basic_params["chemo_sigmoid_betas"] = np.array(
        [chemo_coeff / d_max for i in patient_types],
    )
    basic_params["radio_sigmoid_betas"] = np.array(
        [radio_coeff / d_max for i in patient_types],
    )
    
    if toxicity:
        kg = np.array(
            [np.random.normal(0.00014,0.00001) for i in patient_types])
        #chemotherapy
        kl1 = np.array(
            [np.random.normal(0.001775,0.0001) for i in patient_types])
        
        #radiotherapy
        kl2 = np.array(
            [np.random.normal(0.004125,0.0002) for i in patient_types])
        #colon cancer effect
        kl3 = np.array(
            [np.random.normal(0.000031625,0.000015) for i in patient_types])
        
        kl1_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
        kl2_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])
        kl3_mean_adjustments = np.array([0.0 if i != 2 else 0.1 for i in patient_types])
        
        basic_params['kl1'] = kl1 + kl1 * kl1_mean_adjustments#*norm_rvs1
        basic_params['kl2'] = kl2 + kl2 * kl2_mean_adjustments#*norm_rvs2
        basic_params['kl3'] = kl3 + kl3 * kl3_mean_adjustments#*norm_rvs3
        basic_params['kg'] = kg# + kg * kg_mean_adjustments*norm_rvskg
        
        basic_params['initial_toxicity']  = np.array(
            [np.random.lognormal(mean=np.log(100),sigma=0.2) for i in patient_types])
        
        
    return basic_params

def get_standard_params(num_patients):  # additional params
    """
    Simulation parameters from the Nature article + adjustments for static variables

    :param num_patients:
    :return: simulation_parameters
    """
    
    # Adjustments for static variables
    possible_patient_types = [1, 2, 3]
    patient_types = np.random.choice(possible_patient_types, num_patients)
    chemo_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
    radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])
    
    total = 0
    for k in cancer_stage_observations:
        total += cancer_stage_observations[k]
    cancer_stage_proportions = {
        k: float(cancer_stage_observations[k]) / float(total)
        for k in cancer_stage_observations
    }

    # remove possible entries
    possible_stages = list(tumour_size_distributions.keys())
    possible_stages.sort()
    possible_stages=possible_stages


    initial_stages = np.random.choice(
        possible_stages,
        num_patients,
        p=[cancer_stage_proportions[k] for k in possible_stages],
    )

    # Get info on patient stages and initial volumes
    output_initial_diam = []
    patient_sim_stages = []
    for stg in possible_stages:
        count = np.sum((initial_stages == stg) * 1)

        mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]

        # Convert lognorm bounds in to standard normal bounds
        lower_bound = (np.log(lower_bound) - mu) / sigma
        upper_bound = (np.log(upper_bound) - mu) / sigma

        logging.info(
            (
                "Simulating initial volumes for stage {} "
                + " with norm params: mu={}, sigma={}, lb={}, ub={}"
            ).format(stg, mu, sigma, lower_bound, upper_bound),
        )

        norm_rvs = truncnorm.rvs(
            lower_bound,
            upper_bound,
            size=count,
        )  # truncated normal for realistic clinical outcome

        initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
        output_initial_diam += list(initial_volume_by_stage)
        patient_sim_stages += [stg for i in range(count)]

    # Fixed params
    K = calc_volume(30)  # carrying capacity given in cm, so convert to volume
    alpha_beta_ratio = 10
    alpha_rho_corr = 0.87

    # Distributional parameters for dynamics
    parameter_lower_bound = 0.0
    parameter_upper_bound = 0.3 #Grenze angepasst
    rho_params = (7 * 10**-5, 7.23 * 10**-3)
    alpha_params = (0.0398, 0.088)
    beta_c_params = (0.02, 0.0007)

    # Get correlated simulation paramters (alpha, beta, rho) which respects bounds
    alpha_rho_cov = np.array(
        [
            [alpha_params[1] ** 2, alpha_rho_corr * alpha_params[1] * rho_params[1]],
            [alpha_rho_corr * alpha_params[1] * rho_params[1], rho_params[1] ** 2],
        ],
    )

    alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])

    simulated_params = []

    while (
        len(simulated_params) < num_patients
    ):  # Keep on simulating till we get the right number of params

        param_holder = np.random.multivariate_normal(
            alpha_rho_mean,
            alpha_rho_cov,
            size=num_patients,
        )

        for i in range(param_holder.shape[0]):

            # Ensure that all params fulfill conditions
            if (
                param_holder[i, 0] > parameter_lower_bound
                and param_holder[i, 1] > parameter_lower_bound
                and param_holder[i, 0] < parameter_upper_bound
                and param_holder[i, 1] < parameter_upper_bound
            ):
                simulated_params.append(param_holder[i, :])

        logging.info(
            "Got correlated params for {} patients".format(len(simulated_params)),
        )

    simulated_params = np.array(simulated_params)[
        :num_patients, :
    ]  # shorten this back to normal
    alpha_adjustments = alpha_params[0] * radio_mean_adjustments
    alpha = simulated_params[:, 0]*0.75 + alpha_adjustments
    rho = simulated_params[:, 1]
    beta = alpha / alpha_beta_ratio
    
    # Get the remaining indep params
    logging.info("Simulating beta c parameters")
    beta_c_adjustments = beta_c_params[0] * chemo_mean_adjustments
    beta_c = (
        beta_c_params[0]*0.75
        + beta_c_params[1]
        * truncnorm.rvs(
            (parameter_lower_bound - beta_c_params[0]) / beta_c_params[1],
            (parameter_upper_bound - beta_c_params[0]) / beta_c_params[1],
            size=num_patients,
        )
        + beta_c_adjustments
    )

    output_holder = {
        "patient_types": patient_types,
        "initial_stages": np.array(patient_sim_stages),
        "initial_volumes": calc_volume(
            np.array(output_initial_diam),
        ),  # assumed spherical with diam
        "alpha": alpha,
        "rho": rho,
        "beta": beta,
        "beta_c": beta_c,
        "K": np.array([K for i in range(num_patients)]),
    }

    # Randomise output params
    logging.info("Randomising outputs")
    idx = [i for i in range(num_patients)]
    np.random.shuffle(idx)

    output_params = {}
    for k in output_holder:
        output_params[k] = output_holder[k][idx]
    
    return output_params


def simulate(simulation_params, num_time_steps, assigned_actions=None, toxicity=False, continuous_therapy=False):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :param toxicity: describes, whether weight loss is modelled or not
    :param continuous_therapy: describes, whether dosages or only binary decisions are modelled
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]
    
    if continuous_therapy:
        chemo_amt=[3,5,7]
        radio_amt=[1,2,3]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params["initial_stages"]
    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params[
        "window_size"
    ]  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]
    
    

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]
        toxic = np.zeros((num_patients, num_time_steps))
        noise_terms_toxic = 0.0015 * np.random.randn(
            num_patients,
            num_time_steps,
        )

    noise_terms = 0.01 * np.random.randn(
        num_patients,
        num_time_steps,
    )  # 5% cell variability
    
    recovery_rvs = np.random.rand(num_patients, num_time_steps)

    # Run actual simulation
    for i in range(num_patients):
        if i % 200 == 0:
            logging.info("Simulating patient {} of {}".format(i, num_patients))
        noise = noise_terms[i]
        

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        if toxicity:
            noise_toxic = noise_terms_toxic[i]
            toxic[i, 0] = initial_toxicity[i]
            kg=kgs[i]
            kl1=kl1s[i]
            kl2=kl2s[i]
            kl3=kl3s[i]

        for t in range(0, num_time_steps - 1):

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t - 1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used],
            ).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            if assigned_actions is not None:
                chemo_prob = assigned_actions[i, t, 0]
                radio_prob = assigned_actions[i, t, 1]
            else:
                
                if continuous_therapy:
                    # using softmax
                    chemo_prob=softmax([j*chemo_sigmoid_betas[i]*(cancer_metric_used - chemo_sigmoid_intercepts[i]) for j in range(len(chemo_amt)+1)])
                    radio_prob=softmax([j*radio_sigmoid_betas[i]*(cancer_metric_used - radio_sigmoid_intercepts[i]) for j in range(len(radio_amt)+1)])
                    
                else:
                
                    radio_prob = 1.0 / (
                        1.0
                        + np.exp(
                            -radio_sigmoid_betas[i]
                            * (cancer_metric_used - radio_sigmoid_intercepts[i]),
                        )
                    )
                    chemo_prob = 1.0 / (
                        1.0
                        + np.exp(
                            -chemo_sigmoid_betas[i]
                            * (cancer_metric_used - chemo_sigmoid_intercepts[i]),
                        )
                    )
                
                    chemo_probabilities[i, t] = chemo_prob
                    radio_probabilities[i, t] = radio_prob
            
            # Action application
            if continuous_therapy:
                radio_dosage[i, t] = np.random.choice([0]+radio_amt,p=radio_prob)   
                radio_application_point[i, t] = radio_dosage[i,t]
            else:
                radio_application_point[i, t] = 1
                radio_dosage[i, t] = radio_amt[0]
                
            if continuous_therapy:
                current_chemo_dose = np.random.choice([0]+chemo_amt,p=chemo_prob)   
                chemo_application_point[i, t] = current_chemo_dose
            else:
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]
                    
            # Update chemo dosage
            chemo_dosage[i, t] = (
                previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
                + current_chemo_dose
            )

            cancer_volume[i, t + 1] = cancer_volume[i, t] * (
                1
                + rho * np.log(K / cancer_volume[i, t])
                - beta_c * chemo_dosage[i, t]
                - (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)
                + noise[t]
            )  # add noise to fit residuals
                
            if cancer_volume[i, t + 1] > tumour_death_threshold:
                #print(i)
                cancer_volume[i, t + 1] = tumour_death_threshold
                break  # patient death

            if recovery_rvs[i, t + 1]/50 > cancer_volume[i, t + 1]: 
                cancer_volume[i, t + 1] = 0
                break
            
            if toxicity:
                toxic[i, t + 1] = toxic[i, t] * (
                    1
                    + kg * toxic[i, t] *(1-toxic[i,t]/(initial_toxicity[i])) 
                    - kl1 * chemo_dosage[i, t]
                    - kl2 * radio_dosage[i, t]
                    - kl3 * cancer_volume[i, t]
                    + noise_toxic[t]
                )


        # Package outputs
        sequence_lengths[i] = int(t + 1)
    
    if toxicity:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": chemo_dosage,
            "radio_dosage": radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": chemo_probabilities,
            "radio_probabilities": radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
            "toxicity": toxic
        }
        
    else:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": chemo_dosage,
            "radio_dosage": radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": chemo_probabilities,
            "radio_probabilities": radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
        }

    return outputs

def get_scaling_params(sim, toxicity=False,continuous=False):
    real_idx = ["cancer_volume", "chemo_dosage", "radio_dosage"]
    if toxicity:
        real_idx.append("toxicity")
    if continuous:
        real_idx.append("chemo_application")
        real_idx.append("radio_application")

    means = {}
    stds = {}
    seq_lengths = sim["sequence_lengths"]
    for k in real_idx:
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            active_values += list(sim[k][i, :end])

        means[k] = np.mean(active_values)
        stds[k] = np.std(active_values)

    # Add means for static variables`
    means["patient_types"] = np.mean(sim["patient_types"])
    stds["patient_types"] = np.std(sim["patient_types"])

    return pd.Series(means), pd.Series(stds)

def get_cancer_sim_data(
    chemo_coeff,
    radio_coeff,
    b_load,
    b_save=False,
    seed=100,
    model_root="results",
    window_size=15,
    num_time_steps=60,
    num_patients=10000,
    max_horizon=5,
    toxicity=False,
    num_patients_test=1000,
    continuous_therapy=False
):
    if window_size == 15:
        pickle_file = os.path.join(
            model_root,
            "new_cancer_sim_{}_{}.p".format(chemo_coeff, radio_coeff),
        )
    else:
        pickle_file = os.path.join(
            model_root,
            "new_cancer_sim_{}_{}_{}.p".format(chemo_coeff, radio_coeff, window_size),
        )

    def _generate():
        np.random.seed(seed)

        params = get_confounding_params(
            num_patients,
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            toxicity=toxicity
        )
        params["window_size"] = window_size
        training_data = simulate(params, num_time_steps, toxicity=toxicity,continuous_therapy=continuous_therapy)

        params = get_confounding_params(
            int(num_patients_test),
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            toxicity=toxicity
        )
        params["window_size"] = window_size
        validation_data = simulate(params, num_time_steps,toxicity=toxicity,continuous_therapy=continuous_therapy)

        params = get_confounding_params(
            int(num_patients_test),
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            toxicity=toxicity
        )
        params["window_size"] = window_size
        test_data_factuals = simulate(params, num_time_steps,toxicity=toxicity,continuous_therapy=continuous_therapy)
        
        #validation_data=None
        #test_data_factuals=None

        params = get_confounding_params(
            int(num_patients),
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            toxicity=toxicity
        )
        params["window_size"] = window_size

        scaling_data = get_scaling_params(training_data, toxicity=toxicity,continuous=continuous_therapy)
        
        pickle_map = {
            "chemo_coeff": chemo_coeff,
            "radio_coeff": radio_coeff,
            "num_time_steps": num_time_steps,
            "training_data": training_data,
            "validation_data": validation_data,
            "test_data_factuals": test_data_factuals,
            "scaling_data": scaling_data,
            "window_size": window_size,
        }

        if b_save:
            logging.info("Saving pickle map to {}".format(pickle_file))
            pickle.dump(pickle_map, open(pickle_file, "wb"))
        return pickle_map

    # Controls whether to regenerate the data, or load from a persisted file
    if not b_load:
        pickle_map = _generate()

    else:
        logging.info("Loading pickle map from {}".format(pickle_file))

        try:
            pickle_map = pickle.load(open(pickle_file, "rb"))

        except IOError:
            logging.info(
                "Pickle file does not exist, regenerating: {}".format(pickle_file),
            )
            pickle_map = _generate()

    return pickle_map
