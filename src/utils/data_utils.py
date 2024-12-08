# Adaptation from main.py from https://github.com/seedatnabeel/TE-CDE/blob/main/main.py
# Nabeel Seedat et al. “Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations”. In: Proceedings of the 39th International Conference on Machine Learning. Ed. by Kamalika Chaudhuri et al. Vol. 162.
# Proceedings of Machine Learning Research. PMLR, July 2022, pp. 19497–19521

import pickle

import numpy as np

def get_processed_data(raw_sim_data, scaling_params, toxicity=False, continuous=False):
    """
    It takes the raw simulation data and the scaling parameters, and returns a dictionary with the
    following keys:

    - `current_covariates`: the current covariates (cancer volume and patient type)
    - `time_covariates`: the time covariates (intensity)
    - `previous_treatments`: the previous treatments (one-hot encoded)
    - `current_treatments`: the current treatments (one-hot encoded)
    - `outputs`: the outputs (cancer volume)
    - `active_entries`: the active entries (1 if the patient is still alive, 0 otherwise)
    - `unscaled_outputs`: the unscaled outputs (cancer volume)
    - `input_means`: the input means (cancer volume, patient type, chemo application, radio application)
    - `inputs_stds`: the input standard deviations (cancer volume,

    raw_sim_data (dict): the dataframe containing the simulation data
    scaling_params (tuple): the mean and standard deviation of the cancer volume and patient types
    toxicity (boolean): describes, whether weight loss is modelled or not
    continuous (boolean): describes, whether dosages or binary treatment are modelled

    CODE ADAPTED FROM: https://github.com/ioanabica/Counterfactual-Recurrent-Network
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    if not continuous:

        mean["chemo_application"] = 0
        mean["radio_application"] = 0
        std["chemo_application"] = 1
        std["radio_application"] = 1
    
    if toxicity:
        input_means = mean[
            ["cancer_volume", "patient_types", "chemo_application", "radio_application","toxicity"]
        ].values.flatten()
        input_stds = std[
            ["cancer_volume", "patient_types", "chemo_application", "radio_application","toxicity"]
        ].values.flatten()
    
        # Continuous values
        toxic= (raw_sim_data["toxicity"] - mean["toxicity"]) / std[
            "toxicity"
        ]
    else:
    
        input_means = mean[
            ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
        ].values.flatten()
        input_stds = std[
            ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
        ].values.flatten()
    
        # Continuous values
    cancer_volume = (raw_sim_data["cancer_volume"] - mean["cancer_volume"]) / std[
        "cancer_volume"
    ]
    patient_types = (raw_sim_data["patient_types"] - mean["patient_types"]) / std[
        "patient_types"
    ]

    patient_types = np.stack(
        [patient_types for t in range(cancer_volume.shape[1])],
        axis=1,
    )

    # Binary application
    chemo_application = raw_sim_data["chemo_application"]
    radio_application = raw_sim_data["radio_application"]
    sequence_lengths = raw_sim_data["sequence_lengths"]

    if continuous:
        unscaled_treatments = np.concatenate(
            [
                chemo_application[:, :-offset, np.newaxis],
                radio_application[:, :-offset, np.newaxis],
            ],
            axis=-1,
        )
        
        chemo_application= (raw_sim_data["chemo_application"] - mean["chemo_application"]) / std[
            "chemo_application"
        ]
        radio_application= (raw_sim_data["radio_application"] - mean["radio_application"]) / std[
            "radio_application"
        ]
    
    treatments = np.concatenate(
        [
            chemo_application[:, :-offset, np.newaxis],
            radio_application[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )

    if continuous:
        previous_treatments = treatments[:, :-1, :]
        unscaled_previous_treatments = unscaled_treatments[:, :-1, :]
    else:
        # Convert treatments to one-hot encoding
        one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
        for patient_id in range(treatments.shape[0]):
            for timestep in range(treatments.shape[1]):
                if (
                    treatments[patient_id][timestep][0] == 0
                    and treatments[patient_id][timestep][1] == 0
                ):
                    one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
                elif (
                    treatments[patient_id][timestep][0] == 1
                    and treatments[patient_id][timestep][1] == 0
                ):
                    one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
                elif (
                    treatments[patient_id][timestep][0] == 0
                    and treatments[patient_id][timestep][1] == 1
                ):
                    one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
                elif (
                    treatments[patient_id][timestep][0] == 1
                    and treatments[patient_id][timestep][1] == 1
                ):
                    one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]
    
        one_hot_previous_treatments = one_hot_treatments[:, :-1, :]
    
    if toxicity:
    
        current_covariates = np.concatenate(
            [
                cancer_volume[:, :-offset, np.newaxis],
                patient_types[:, :-offset, np.newaxis],
                toxic[:, :-offset, np.newaxis]
            ],
            axis=-1,
        )
        outputs_toxic = toxic[:, horizon:, np.newaxis]
        output_toxic_means = mean[["toxicity"]].values.flatten()[
            0
        ]  # because we only need scalars here
        output_toxic_stds = std[["toxicity"]].values.flatten()[0]
        
    else:
        current_covariates = np.concatenate(
            [
                cancer_volume[:, :-offset, np.newaxis],
                patient_types[:, :-offset, np.newaxis],
            ],
            axis=-1,
        )
    
    time_covariates=np.zeros(shape=cancer_volume[:,:-offset,np.newaxis].shape)
    for i in range(len(raw_sim_data["sequence_lengths"])):
        time_covariates[i,:int(raw_sim_data["sequence_lengths"][i]),0]=[j/59 for j in range(int(raw_sim_data["sequence_lengths"][i]))]
    
    outputs = cancer_volume[:, horizon:, np.newaxis]

    output_means = mean[["cancer_volume"]].values.flatten()[
        0
    ]  # because we only need scalars here
    output_stds = std[["cancer_volume"]].values.flatten()[0]

    # Add active entires
    active_entries = np.zeros(outputs.shape)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])
        active_entries[i, :sequence_length, :] = 1

    raw_sim_data["current_covariates"] = current_covariates
    raw_sim_data["time_covariates"] = time_covariates
    if continuous:
        raw_sim_data["previous_treatments"] = previous_treatments
        raw_sim_data["current_treatments"] = treatments
        raw_sim_data["unscaled_previous_treatments"] = unscaled_previous_treatments
        raw_sim_data["unscaled_current_treatments"] = unscaled_treatments
    else:
        raw_sim_data["previous_treatments"] = one_hot_previous_treatments
        raw_sim_data["current_treatments"] = one_hot_treatments
    raw_sim_data["outputs"] = outputs
    raw_sim_data["active_entries"] = active_entries

    raw_sim_data["unscaled_outputs"] = (
        outputs * std["cancer_volume"] + mean["cancer_volume"]
    )
    raw_sim_data["input_means"] = input_means
    raw_sim_data["inputs_stds"] = input_stds
    raw_sim_data["output_means"] = output_means
    raw_sim_data["output_stds"] = output_stds
    
    if toxicity:
        #raw_sim_data["toxicity"] = toxic
        raw_sim_data["unscaled_outputs_toxicity"] = (
            outputs_toxic * std["toxicity"] + mean["toxicity"]
        )
        raw_sim_data["outputs_toxicity"]=outputs_toxic
        raw_sim_data["output_toxicity_means"] = output_toxic_means
        raw_sim_data["output_toxicity_stds"] = output_toxic_stds

    return raw_sim_data


def process_data(pickle_map,toxicity=False,continuous=False,treatment_testdata=False, scaling_data=None):
    """
    Returns processed train, val, test data from pickle map

    Args:
    pickle_map (dict): dict containing data from pickle map
    toxicity (boolean): describes, whether weight loss is modelled or not
    continuous (boolean): describes, whether dosages or binary treatment are modelled
    treatment_test_data (boolean): Only simulating test data, default false
    scaling_data (dict): Dict containing scaling parameters
    
    Returns:
    training_processed (np array): training data processed numpy
    validation_processed (np array): validation data processed numpy
    test_processed (np array): test data processed numpy
    """
    if treatment_testdata:
        test_processed = get_processed_data(pickle_map, scaling_data, toxicity=toxicity, continuous=continuous)
    
        return test_processed
    else:
        # load data from pickle_map
        training_data = pickle_map["training_data"]
        validation_data = pickle_map["validation_data"]
        test_data = pickle_map["test_data_factuals"]
        scaling_data = pickle_map["scaling_data"]
    
        # get processed data
        training_processed = get_processed_data(training_data, scaling_data, toxicity=toxicity, continuous=continuous)
        validation_processed = get_processed_data(validation_data, scaling_data, toxicity=toxicity, continuous=continuous)
        test_processed = get_processed_data(test_data, scaling_data, toxicity=toxicity, continuous=continuous)
    
        return training_processed, validation_processed, test_processed


def read_from_file(filename):
    """
    It loads the file from pickle.

    filename (str): the name of the file to read from
    return: A list of dictionaries.
    """
    # load file from pickle

    return pickle.load(open(filename, "rb"))
