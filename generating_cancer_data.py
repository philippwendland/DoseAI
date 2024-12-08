# Adaptation from main.py from https://github.com/seedatnabeel/TE-CDE/blob/main/main.py
# Nabeel Seedat et al. “Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations”. In: Proceedings of the 39th International Conference on Machine Learning. Ed. by Kamalika Chaudhuri et al. Vol. 162.
# Proceedings of Machine Learning Research. PMLR, July 2022, pp. 19497–19521

import argparse
import logging
import os

from src.utils.cancer_simulation import get_cancer_sim_data
from src.utils.data_utils import read_from_file, write_to_file

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=4, type=int)
    parser.add_argument("--radio_coeff", default=4, type=int)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_name", default="te_cde_test")
    parser.add_argument("--load_dataset", default=True)
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--use_transformed", default=True)
    parser.add_argument("--multistep", default=False)
    parser.add_argument("--kappa", type=int, default=10)
    parser.add_argument("--lambda_val", type=float, default=1)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_horizon", type=int, default=5)
    parser.add_argument("--save_raw_datapath", type=str, default=None)
    parser.add_argument("--save_transformed_datapath", type=str, default=None)
    parser.add_argument("--hidden_channels_x", type=int, default=8)
    parser.add_argument("--output_channels", type=int, default=59)
    parser.add_argument("--sample_proportion", type=int, default=1)
    parser.add_argument("--use_time", default=False)    
    parser.add_argument("--epochs", type=int, default=100) 
    parser.add_argument("--batch_size",type=int, default=512) 
    parser.add_argument("--patience", type=int, default=5) 
    parser.add_argument("--hidden_states_x", type=int, default=128) 
    parser.add_argument("--delta",type=int,default=0.0001)
    parser.add_argument("--lr",type=int,default=0.0001)
    parser.add_argument("--momentum",type=int,default=0.9)
    parser.add_argument("--multistep_epochs",type=int,default=10)
    parser.add_argument("--treatment_options",type=int,default=4)
    parser.add_argument("--dropout_rate",type=int,default=0.1)
    parser.add_argument("--window_size",type=int,default=15)
    parser.add_argument("--num_time_steps",type=int,default=60)
    parser.add_argument("--num_patients",type=int,default=1000)
    parser.add_argument("--seq_length",type=int,default=4)
    parser.add_argument("--toxicity",default=False)
    parser.add_argument("--mcd", default=True)
    parser.add_argument("--factual",default=False)
    parser.add_argument("--hidden_channels_x_multi", type=int, default=8)
    parser.add_argument("--hidden_states_x_multi", type=int, default=128) 
    parser.add_argument("--batch_size_multi",type=int, default=512) 
    parser.add_argument("--lr_multi",type=int,default=0.0001)
    parser.add_argument("--dropout_rate_multi",type=int,default=0.1)
    return parser.parse_args()

#This part of the script makes it run in spyder
if __name__ == "__main__":

    args = init_arg()
    #Defining parameters
    args.chemo_coeff=4
    args.radio_coeff=4
    args.max_horizon=5
    args.multistep=True
    args.save_raw_datapath="C:/Users/wendland/Documents/GitHub/TE-CDE-main"
    args.save_transformed_datapath="C:/Users/wendland/Documents/GitHub/TE-CDE-main"
    args.use_transformed=True
    #args.data_path="C:/Users/wendland/Documents/GitHub/TE-CDE-main/new_cancer_sim_4_4_kappa_10_toxicity.p"
    args.hidden_channels_x = 8
    args.output_channels = 59
    args.sample_proportion = 1
    args.use_time = False
    args.epochs=100
    args.batch_size=512
    args.patience=5
    args.delta=0.0001
    args.lr=0.0001
    args.momentum=0.9
    args.multistep_epochs=10
    args.treatment_options=4
    args.dropout_rate=0.1
    args.window_size=15
    args.num_time_steps=24
    args.num_patients=1000
    args.seq_length=4
    args.toxicity=True
    args.mcd = False
    args.hidden_channels_x_multi=8
    args.hidden_states_x_multi=128
    args.dropout_rate_multi=0.1
    args.lr_multi=0.0001
        
    
    if not os.path.exists("./tmp_models/"):
        os.mkdir("./tmp_models/")

    use_transformed = str(args.use_transformed) == "True"
    multistep = str(args.multistep) == "True"
    
    strategy = "all"
    
    args.toxicity=True
    continuous_therapy=True
    if args.data_path == None:
        logging.info("Generating dataset")

        continuous_therapy=True
        pickle_map = get_cancer_sim_data(
            chemo_coeff=args.chemo_coeff,
            radio_coeff=args.radio_coeff,
            b_load=True,
            b_save=False,
            model_root=args.results_dir,
            window_size=args.window_size,
            num_time_steps=args.num_time_steps,
            num_patients=args.num_patients,
            max_horizon=args.max_horizon,
            toxicity=args.toxicity,
            continuous_therapy=continuous_therapy,
            seed=100
        )

    else:
        logging.info(f"Loading dataset from: {args.data_path}")
        pickle_map = read_from_file(args.data_path)
    
   
    coeff = int(args.radio_coeff)

    if args.save_raw_datapath != None:
        logging.info(f"Writing raw data to {args.save_raw_datapath}")
        write_to_file(
            pickle_map,
            f"{args.save_raw_datapath}/test_cancer_sim_{coeff}_{coeff}.p",
            #f"{args.save_raw_datapath}/new_cancer_sim___{coeff}_{coeff}.p",
        )
        
    if args.save_transformed_datapath != None:
        logging.info(f"Writing transformed data to {args.save_transformed_datapath}")
        write_to_file(
            pickle_map,
            f"{args.save_transformed_datapath}/data_dict.p",
        )
