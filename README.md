# DoseAI

This Github repository encompasses the implementation of "DoseAI" as presented by Wendland and Kschischo [1].
DoseAI is a completely data-driven online-updateable dose optimization model based on Artificial Intelligence accounting for side-effects. 
DoseAI is able to handle the special characteristics of patient data including irregular measurements, numerous missing values and time-dependent confounding. 

We performed an iterative optimal chemotherapy and radiotherapy dosing regimens optimization for cancer patients aimed at minimizing the tumor volume while accounting for the weight loss as a severe side-effect.
Our code builds upon the Neural CDE implementation from [2], the Treatment-Effect Controlled differential equation [3], the treatment optimization model OptAB for Sepsis patients [4] and on the data generation model[5,6]. 

# Contents of the Repository

* Scripts for generating synthetic patient data using a PK-PD model based on [7] and [8]
* Scripts for executing DoseAI.
* Scripts to create the plots showcased in our paper "DoseAI" located in the "code" folder.
* The final Encoder and Decoder torch models of DoseAI.
* The package dependencies to run DoseAI are located in the Code/DoseAI folder.

# How to run DoseAI

* To generate synthetic patient data use the script generating_cancer_data.py
* To conduct hyperparameteroptimization (and training) execute the hypopt_encoder_cancer.py and subsequently the hypopt_decoder_cancer.py file.
* For optimal treatment selection utilize the script treatment_optimization_pymoo.
* To create the plots of our paper run the plot functions.

Remark: Comprehensive descriptions of the code can be found in the utils_paper.py file.

# References

[1] Wendland, P. & Kschischo, M. "DoseAI" presented in Wendland and Kschischo. (2024)

[2] Kidger, P., Morrill, J., Foster, J. & Lyons, T. "Neural Controlled Differential Equations for Irregular Time Series". In Advances in Neural Information Processing Systems, vol. 33 (Curran Associates, Inc., 2020)

[3] Seedat, N., Imrie, F., Bellot, A. & Qian, Z. "Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations". In Proceedings of the 39th International Conference on Machine Learning, vol. 162 of Proceedings of Machine Learning Research,19497–19521 (PMLR, 2022)

[4] Wendland, P., Schenkel-Häger, C., Wenningmann, I. & Kschischo, M. "An optimal antibiotic selection framework for Sepsis patients using Artificial Intelligence". In NPJ Digital Medicine, vol. 7.1, p.343. ISSN: 2398-6352. DOI: 10.1038/s41746-024-01350-y (2024)

[5] Bica, I., Alaa, A., Jordon, J. & van der Schaar, M. "Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations". In 8th International Conference on Learning Representation (ICLR, 2020)

[6] Lim, B., Alaa, A. & van der Schaar, M. "Forecasting Treatment Responses Over Time Using Recurrent Marginal Structural Networks". In Advances in Neural Information Processing Systems, vol. 31 (Curran Associates, Inc., 2018)

[7] Geng, C., Paganett, H. & Grassberger, C. "Prediction of treatment response for combined chemo-and radiation therapy for non-small cell lung cancer patients using a bio-mathematical model". In Scientific Reports, vol. 7.1, p.13542, ISSN: 0090-3493, DOI: 10.1038/s41598-017-13646-z (2017)

[8] Hadjiandrou, M., & Mitsis, G. "Mathematical Modeling of Tumor Growth, Drug-Resistance, Toxicity, and Optimal Therapy Design". In IEEE Transactions on Biomedical Engineering, vol. 61.2, pp. 415–425. ISSN: 0018-9294, 1558-2531. DOI: 10.1109/TBME.2013.2280189 (2014)

