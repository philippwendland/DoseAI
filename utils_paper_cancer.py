import torch
import torchcde
import numpy as np
import torchmetrics
import torch.nn as nn
import torchsort
import copy
import matplotlib.pyplot as plt
import pickle

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_states, activation = 'relu',num_depth=1):
        ######################
        # input_channels is the number of input channels in the data Z. (Determined by the data.)
        # hidden_channels is the number of channels for F. (Determined by you!)
        # hidden_states is the maximum number of units for F
        # activation is the activation function of F
        # num_depth is the number of layers for F
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        if activation == 'leakyrelu':
            self.activation=nn.LeakyReLU()
        elif activation =='tanh':
            self.activation=nn.Tanh()
        elif activation =='relu':
            self.activation=nn.ReLU()
        elif activation =='sigmoid':
            self.activation=nn.Sigmoid()
        elif activation =='identity':
            self.activation=nn.Identity()
        
        diff = hidden_states-hidden_channels
        diff2 = hidden_states-input_channels*hidden_channels
        layers=[]
        for i in range(num_depth):
            layers.append(torch.nn.Linear(hidden_channels+int(np.round((diff*i/num_depth))),hidden_channels+int(np.round((diff*(i+1)/num_depth)))))
            layers.append(self.activation)    
        for i in range(num_depth):
            layers.append(torch.nn.Linear(hidden_channels*input_channels+int(np.round((diff2*(num_depth-i)/num_depth))),hidden_channels*input_channels+int(np.round((diff2*(num_depth-i-1)/num_depth)))))
            layers.append(self.activation)
        self.net = nn.Sequential(*layers)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.net(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_states=128, output_channels=2, treatment_options=4, activation='leakyrelu', num_depth=1, z0_dimension_dec=None, interpolation="linear", device=None, continuous=False, treat_thresh=None, pos=False, thresh=None, pred_comp=False, pred_act=None, pred_states=128, pred_depth=1,static_dim=0):
   
        super(NeuralCDE, self).__init__()
        
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for F. (Determined by you!)
        # hidden_states is the maximum number of units for F
        # activation is the activation function of F
        # num_depth is the number of layers for F
        
        # output_channels: Number of outputs
        # treatment_options: Number of treatments
        # zo_dimension_dec: Dimension of the initial value of the decoder (Determined by the data and the Encoder)
        # interpolation: Indicator whether cubic or (recti)linear interpolation is used
        # device: Optional variable indicating GPU or CPU usage
        # pos and thresh: Indicate whether only positive values can be predicted or not using softplus. 
        # thresh corresponds to the unnormalized 0 if data are normalized
        # pred_comp=False: Indicates whether a deep neural network is used for computing the output or not
        # pred_act=None, pred_states=128, pred_depth=1: Optional parameters describing the activation function of,
        # maximum number of units, and the number of layers of h
        # static_dim: Corresponds to the dimension of the static variables (determined by the data)
        # continuous: Boolean Describing whether treatment is modelled continuous or not
        # treat_thresh: If continuous treatment, corresponds to lower treatment threshold (often zero)

        # function F of the CDE
        self.func = CDEFunc(input_channels, hidden_channels, hidden_states, activation, num_depth)
        # computing initial values
        if z0_dimension_dec is not None:
            self.initial = torch.nn.Linear(z0_dimension_dec, hidden_channels)
        else:
            self.initial = torch.nn.Linear(input_channels+static_dim, hidden_channels)
        self.pred_comp=pred_comp
        # function for computing the output
        if pred_comp:
            #self.pred_dec1 = torch.nn.Linear(hidden_channels,pred_states)
            if pred_act == 'leakyrelu':
                self.pred_actlayer=nn.LeakyReLU()
            elif pred_act =='tanh':
                self.pred_actlayer=nn.Tanh()
            elif pred_act =='relu':
                self.pred_actlayer=nn.ReLU()
            elif pred_act =='sigmoid':
                self.pred_actlayer=nn.Sigmoid()
            elif pred_act =='identity':
                self.pred_actlayer=nn.Identity()
            #self.pred_dec2 = torch.nn.Linear(pred_states, output_channels)
                
            diff = pred_states-hidden_channels
            diff2 = pred_states-output_channels
            layers=[]
            for i in range(pred_depth):
                layers.append(torch.nn.Linear(hidden_channels+int(np.round((diff*i/pred_depth))),hidden_channels+int(np.round((diff*(i+1)/pred_depth)))))
                layers.append(self.pred_actlayer)    
            for i in range(pred_depth):
                layers.append(torch.nn.Linear(output_channels+int(np.round((diff2*(pred_depth-i)/pred_depth))),output_channels+int(np.round((diff2*(pred_depth-i-1)/pred_depth)))))
                if i < pred_depth-1:
                    layers.append(self.pred_actlayer)
            self.pred_dec = nn.Sequential(*layers)
        #else:
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        self.softplus = torch.nn.Softplus()
        if continuous:
            self.treat_thresh=treat_thresh
        else:
            self.treatment_act = torch.nn.Softmax(dim=1)
        self.continuous = continuous
        self.output_channels=output_channels
        self.treatment = torch.nn.Linear(hidden_channels, treatment_options)
        
        if device=='cpu':
            device_type="cpu"
        elif device=='cuda' or device=='gpu':
            device_type="cuda"
        else:
            if torch.cuda.is_available():
                device_type = "cuda"
            else:
                device_type = "cpu"
        self.device = torch.device(device_type)
        self.treatment_options=treatment_options
        self.pos=pos
        self.thresh=thresh

    def forward(self, coeffs, max_T=None, z0=None, static=None):
        # input: 
        # coeffs: Interpolated coefficients for Neural CDE
        # max_T: optional, Maximal prediction time of the Neural CDE (only necessary for linear interpolation/extrapolation), default: None
        # z0: optional, Initial values for the Decoder of the Neural CDE, default None
        # static: optional, Static values to be concatenated for the Encoder of the Neural CDE, default None
        
        # output: 
        # pred_y_tens: tensor of outcomepredictions, size: batchsize x timepoints x 1 (dim of output)
        # pred_a: tensor of treatment predictions, size: batchsize x timepoints x number of side-effects
        # z_T: latent state of the Neural CDE, size: batchsize x timepoints x hidden_channels
        
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
            
            # complicated, index 2 is first tp, index 3 is second tp, index 4 is third tp etc.
            #t=torch.tensor([i for i in range(batch_covariables[:,:10,:].shape[1])]).float().to(X0.device)
        elif self.interpolation == 'linear':
            
            # Complicated: Setting the starting timepoint for Neural CDE
            # Due to 1-hour predictions, Neural CDE should start with a dynamic for the first prediction.
            # Therefore, the first timepoint is set to 0.5
            
            # Neural CDE returns NaN if only one timepoint is submitted, so a second timepoint is created in the training function using NaNs.
            # First, the covariables are extrapolated by using the linear interpolation
            # So it is necessary to set the time channel to the timepoint of the prediction - 1

            X = torchcde.LinearInterpolation(coeffs.float())
            
            if max_T is not None:
                t=np.array([i*2 for i in range(max_T)],dtype=float)
                t[0]=0.5
                t=torch.from_numpy(np.insert(t,0,0,axis=0)).float().to(self.device)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        
        # Computing initial value including static variables
        X0 = X.evaluate(X.interval[0]).float()
        if static is not None:
            if len(static.shape)==3:
                X0 = torch.cat([X0,static[:,0,:]],dim=1)
            else:
                X0 = torch.cat([X0,static],dim=1)
        
        # z0 for Decoder, X0 for Encoder
        if z0 is None:
            z0 = self.initial(X0).float()
        else:
            z0 = self.initial(z0).float()
        # max_T+1, due to the initial value at start
        ######################
        # Actually solve the CDE.
        ######################

        if self.interpolation=='cubic':
            z_T = torchcde.cdeint(X=X,
                                  z0=z0,
                                  func=self.func,
                                  t=X.interval)
        elif self.interpolation=='linear':
            # alternative
            # z_T = torchcde.cdeint(X=X,
            #                       z0=z0,
            #                       func=self.func,
            #                       t=t,
            #                       method='rk4',
            #                       options=dict(step_size=1))
            
            z_T = torchcde.cdeint(X=X,
                                  z0=z0,
                                  func=self.func,
                                  t=t,
                                  options=dict(jump_t=X.grid_points))
        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1:,:]
        
        # Creating lists and computing the Output based on the latent values with further neural networks
        pred_y_tens = torch.empty(size=[z0.shape[0],z_T.shape[1],self.output_channels], device=X0.device)
        pred_a = torch.empty(size=[z0.shape[0],z_T.shape[1],self.treatment_options],device=X0.device)
        for i in range(z_T.shape[1]):
            if self.pred_comp:
                pred_y_tens[:,i,:] = self.pred_dec(z_T[:,i,:])
            else:
                pred_y_tens[:,i,:] = self.readout(z_T[:,i,:])
            if self.continuous:
                pred_a[:,i] = self.treatment(z_T[:,i,:])
            else:
                pred_a[:,i] = self.treatment_act(self.treatment(z_T[:,i,:]))
        if self.pos:
            for i in range(self.output_channels):
                pred_y_tens[:,:,i]=self.softplus(pred_y_tens[:,:,i]-self.thresh[i])+self.thresh[i]
        if self.continuous:
            for i in range(self.treat_thresh.shape[0]):
                pred_a[:,:,i]=self.softplus(pred_a[:,:,i]-self.treat_thresh[i])+self.treat_thresh[i]
        return pred_y_tens, pred_a, z_T

class EarlyStopping:
    """Early stopping: stop training if validation loss doesn't improve after n patience steps"""

    def __init__(self, patience=5, delta=0.001, path="checkpoint.pt"):
        """
        Args:
            patience (int): Patience in epochs
            delta (float): minimum delta change in validation loss
            path (str): path for model checkpoints
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        if path is None:
            self.path="checkpoint.p"
        else:
            self.path = path
        self.path2 = "checkpoint_multistep.pt"

    def __call__(self, val_loss, model, multistep_model=None):

        score = -val_loss

        # first epoch
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # update counter
        elif score < self.best_score + self.delta or score.isnan().any():
            self.counter += 1
            #print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # save best model
            self.best_score = score
            self.save_checkpoint(val_loss, model, multistep_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, multistep_model=None):
        """Saves model when validation loss decrease."""
        
        torch.save(model.state_dict(), self.path)
        if multistep_model is not None:
            torch.save(multistep_model.state_dict(), self.path2)
        self.val_loss_min = val_loss

def spearman(pred, target, empty_var=False, **kw):
    pred = torchsort.soft_rank(pred.reshape(1,-1),regularization_strength=1, **kw)
    #pred.unique(return_counts=True)
    if empty_var and len(target.unique())==1 and pred.shape[1]>1:
        return 1-(torch.var(pred)/torch.var(torch.arange(pred.shape[1],dtype=float)))
    elif not empty_var and len(target.unique())==1 and pred.shape[1]>1:
        return torch.tensor([np.nan])
    else:
        target = torchsort.soft_rank(target.reshape(1,-1),regularization_strength=1, **kw)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return torch.abs((pred * target).sum())

def prep_map(data_map, device, unscaled= False):
    if unscaled:
        data_tensor=torch.from_numpy(data_map["unscaled_outputs"]).float().to(device)
        toxic_tensor = torch.from_numpy(data_map["unscaled_outputs_toxicity"]).float().to(device)
    else:
        data_tensor=torch.from_numpy(data_map["outputs"]).float().to(device)
    toxic_tensor = torch.from_numpy(data_map["outputs_toxicity"]).float().to(device)
    treatments_tensor = torch.from_numpy(data_map["current_treatments"]).float().to(device)
    covariables_x = torch.from_numpy(np.concatenate([data_map["current_covariates"][:,:-1,:],data_map["previous_treatments"]],axis=2)).float().to(device)
    time_tensor = torch.from_numpy(data_map["time_covariates"]).float().to(device)
    covariables_tensor = torch.cat([covariables_x,time_tensor[:,:-1,:]],dim=2)
    active_tensor=torch.from_numpy(data_map["active_entries"]).float().to(device)
    return data_tensor, toxic_tensor, treatments_tensor, covariables_tensor, time_tensor, active_tensor

def train(model, train_output, train_toxic, train_treatments, covariables, active_entries, validation_output, validation_toxic, validation_treatments, covariables_val, active_entries_val, static = None,static_val = None, lr=0.001, batch_size=500, patience=10, delta=0.0001, max_epochs=1000, weight_loss=True,rectilinear_index=None,early_stop_path=None, mu=None, a_loss='spearman',empty_var=False):
    
    # input: 
    
    # model: Neural CDE model
    # train_output: Tensor of output/treatment success (training data), size: number of patients x timepoints (in hours) x 1 (dim of output)
    # train_toxic: Tensor of the side effects (training data), size: number of patients x timepoints (in hours) x number of side effects
    # train_treatments: Tensor of the treatments (training data), size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables (training data), size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    
    # validation_output: Tensor of output/treatment success (validation data), size: number of patients x timepoints (in hours) x 1 (dim of output)
    # validation_toxic: Tensor of the side effects (validation data), size: number of patients x timepoints (in hours) x number of side effects
    # validation_treatments: Tensor of the treatments (validation data), size: number of patients x timepoints (in hours), x number of treatments
    # covariables_val: Tensor of the covariables (validation data), size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries_val: Boolean Tensor indicating, whether patients is at ICU or discharged (validation data), size: number of patients x timepoints (in hours) x 1
    
    # static: Tensor of static variables (training data): batch x number of variables, default: None
    # static_val: Tensor of static variables (validation data): batch x number of variables, default: None
    
    # lr: learning rate of the optimizer, default: 0.01
    # batch_size, default: 500
    # patience: patience for Early_stopping, default: 10
    # delta: minimum delta change in validation loss, default: 0.0001
    # max_epochs: maximum number of epochs of training, default:1000
    # weight_loss: Indicating whether treatment loss is multiplied by mu or not
    
    # rectilinear_index: Time index of covariables tensor, default: None
    # early_stop_path: Path to save model during Early Stopping, default: None
    # mu: Multiplication factor of treatment loss, default: None
    # a_loss: Categorical Variable corresponding to the loss for treatment
    # empty_var: Help variable indicating, whether there are empty treatment variables or not
    
    # output:
    # loss_val: Validation loss (necessary for hyperparameteroptimization)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    model=model.train()

    early_stopping = EarlyStopping(patience=patience, delta=delta, path=early_stop_path)
    
    # interpolating covariables
    if model.interpolation=='cubic':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables)
        validation_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables_val)
    elif model.interpolation=='linear':
        # default rectilinear index is last index
        if rectilinear_index is None:
            train_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=covariables.shape[2]-1)
            validation_coeffs = torchcde.linear_interpolation_coeffs(covariables_val, rectilinear=covariables_val.shape[2]-1)
        else:
            train_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=rectilinear_index)
            validation_coeffs = torchcde.linear_interpolation_coeffs(covariables_val, rectilinear=rectilinear_index)
    
    # Initialize data loader and loss functions
    if static is not None:
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_output, active_entries, train_toxic, train_treatments, static)
    else:
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_output, active_entries, train_toxic, train_treatments)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
 
    mseloss = torch.nn.MSELoss()
    if not model.continuous:
        treatment_loss = nn.CrossEntropyLoss()
        tl = nn.Softsign()
    
    # Initializing mu
    if mu is None:
        if weight_loss:
            mu = 0
            mu = model.output_channels/model.treatment_options
        else:
            mu=1
    
    for epoch in range(max_epochs):
        for batch in train_dataloader:
            # Setting model to training mode
            model=model.train()
            # Creating batches
            if static is not None:
                batch_coeffs, train_output_b, active_entries_b, train_toxic_b, train_treat_b, static_b = batch
            else:
                batch_coeffs, train_output_b, active_entries_b, train_toxic_b, train_treat_b = batch
                static_b=None
            
            # Computing predictions by the model
            pred_output, pred_a,_ = model(batch_coeffs, max_T=covariables.shape[1],static=static_b)
            
            if model.continuous:
                if a_loss=='spearman':
                    cost=torch.empty(size=[pred_a.shape[1]],device=model.device)
                    for i in range(pred_a.shape[1]):
                        loss=torch.empty(size=[1],device=model.device)
                        loss[:]=np.nan
                        for j in range(pred_a.shape[2]):
                            if j == 0 and len(train_treat_b[:,i+1,j:j+1][active_entries_b[:,i+1,:].bool()])>1:
                                loss = spearman(pred_a[:,i,j:j+1][active_entries_b[:,i+1,:].bool()], train_treat_b[:,i+1,j:j+1][active_entries_b[:,i+1,:].bool()],empty_var=empty_var)
                            elif len(train_treat_b[:,i+1,j:j+1][active_entries_b[:,i+1,:].bool()])>1:
                                loss=loss + spearman(pred_a[:,i,j:j+1][active_entries_b[:,i+1,:].bool()], train_treat_b[:,i+1,j:j+1][active_entries_b[:,i+1,:].bool()],empty_var=empty_var)
                        cost[i] = loss
                        #print(loss)
                    loss_a=torch.nanmean(cost)

            else:
                loss_a_compl = torch.empty([pred_a.shape[1],pred_a.shape[2]],device=model.device)
            
                # Computing treatment loss
                for i in range(pred_a.shape[1]):
                    for j in range(pred_a.shape[2]):
                        loss_a_compl[i,j]=tl(treatment_loss(pred_a[:,i,j][active_entries_b[:,i+1,0].bool()], train_treat_b[:,i+1,j][active_entries_b[:,i+1,0].bool()]))
                loss_a = -torch.nanmean(loss_a_compl)
            if train_output_b.shape[1]==pred_output.shape[1]+1:
                train_output_b=train_output_b[:,:-1,:]
            if train_toxic_b.shape[1]==pred_output.shape[1]+1:
                train_toxic_b=train_toxic_b[:,:-1,:]
            
            # Computing loss of Outputs and Side-effects  
            loss_output=mseloss(pred_output[:,:,0:1][(active_entries_b[:,1:,:].bool()) & (~train_output_b.isnan())], train_output_b[(active_entries_b[:,1:,:].bool()) & (~train_output_b.isnan())])
            loss_toxic_tensor = torch.empty(size=[train_toxic_b.shape[2]],device=model.device)
            for j in range(train_toxic_b.shape[2]):
                loss_toxic_tensor[j] = mseloss((pred_output[:,:,j+1:j+2][(active_entries_b[:,1:,:].bool()) & (~train_toxic_b[:,:,j:j+1].isnan())]), train_toxic_b[:,:,j:j+1][(active_entries_b[:,1:,:].bool()) & (~train_toxic_b[:,:,j:j+1].isnan())])
            loss_toxic = torch.nansum(loss_toxic_tensor)
            
            # Computing training loss
            loss= loss_output + loss_toxic + mu * loss_a
            if loss.isnan():
                return loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(loss_output)
        print(loss_toxic_tensor)
        print(loss_a)
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))
        
        # Setting model to evaluation mode
        model=model.eval()
        
        # Computing predictions of th evalidation data
        pred_output_val, pred_a_val,_ = model(validation_coeffs, max_T=covariables_val.shape[1],static=static_val)
        
        if model.continuous:
            if a_loss=='spearman':
                cost=torch.empty(size=[pred_a_val.shape[1]],device=model.device)
                for i in range(pred_a_val.shape[1]):
                    loss_val=torch.empty(size=[1],device=model.device)
                    loss_val[:]=np.nan
                    for j in range(pred_a_val.shape[2]):
                        if j == 0 and len(validation_treatments[:,i+1,j:j+1][active_entries_val[:,i+1,:].bool()])>1:
                            loss_val = spearman(pred_a_val[:,i,j:j+1][active_entries_val[:,i+1,:].bool()], validation_treatments[:,i+1,j:j+1][active_entries_val[:,i+1,:].bool()],empty_var=empty_var)
                        elif len(validation_treatments[:,i+1,j:j+1][active_entries_val[:,i+1,:].bool()])>1:
                            loss_val=loss_val + spearman(pred_a_val[:,i,j:j+1][active_entries_val[:,i+1,:].bool()], validation_treatments[:,i+1,j:j+1][active_entries_val[:,i+1,:].bool()],empty_var=empty_var)
                        cost[i] = loss_val
                loss_a_val=torch.nanmean(cost)
        else:
            # Initializing and computing validation treatment loss
            loss_a_compl = torch.empty([pred_a_val.shape[1],pred_a_val.shape[2]],device=model.device)
            for i in range(pred_a_val.shape[1]):
                for j in range(pred_a.shape[2]):
                    loss_a_compl[i,j]=tl(treatment_loss(pred_a_val[:,i,j][active_entries_val[:,i+1,0].bool()], validation_treatments[:,i+1,j][active_entries_val[:,i+1,0].bool()]))
        
            loss_a_val = -torch.nanmean(loss_a_compl)
        
        if validation_output.shape[1]==pred_output_val.shape[1]:
            validation_output=validation_output[:,:-1,:]
        if validation_toxic.shape[1]==pred_output_val.shape[1]:
            validation_toxic=validation_toxic[:,:-1,:]
        
        # Computing Output and side-effects loss
        loss_output_val=mseloss(pred_output_val[:,:,0:1][(active_entries_val[:,1:,:].bool()) & (~validation_output.isnan())], validation_output[(active_entries_val[:,1:,:].bool()) & (~validation_output.isnan())])
        
        loss_toxic_tensor_val = torch.empty(size=[validation_toxic.shape[2]],device=model.device)
        for j in range(validation_toxic.shape[2]):
            loss_toxic_tensor_val[j] = mseloss((pred_output_val[:,:,j+1:j+2][(active_entries_val[:,1:,:].bool()) & (~validation_toxic[:,:,j:j+1].isnan())]), validation_toxic[:,:,j:j+1][(active_entries_val[:,1:,:].bool()) & (~validation_toxic[:,:,j:j+1].isnan())])
        loss_toxic_val = torch.nanmean(loss_toxic_tensor_val)
        
        loss_val=loss_output_val + loss_toxic_val + mu * loss_a_val
        if loss_val.isnan():
            return loss_val
        
        print('Epoch: {}   Validation loss: {}'.format(epoch, loss_val.item()))
        
        early_stopping(loss_val, model)
        
        if early_stopping.early_stop:
            break
            print("Early stopping phase initiated...")
        torch.save(model.state_dict(), './test.pth')
                
    torch.save(model.state_dict(), './final_model.pth')
    
    print(loss_output)
    print(loss_toxic_tensor)
    print(loss_a)
    
    return loss_val
    
def predict(model, validation_output, validation_toxic, validation_treatments, covariables, active_entries, static = None, rectilinear_index=None, a_loss='spearman', empty_var=False):
    
    # model: Neural CDE model
    # train_output: Tensor of output/treatment success, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # train_toxic: Tensor of the side effects, size: number of patients x timepoints (in hours) x number of side effects
    # train_treatments: Tensor of the treatments, size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables, size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    # static: Tensor of static variables: number of patients x number of variables, default: None
    # rectilinear_index: Time index of covariables tensor, default: None
    # a_loss: Categorical Variable corresponding to the loss for treatment
    # empty_var: Help variable indicating, whether there are empty treatment variables or not
        
    # output:
    # pred_output: Tensor of the predicted outcomes, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # pred_a: Tensor of the predicted side-effects associated laboratory variables size: number of patients x timepoints (in hours) x number of side effects
    # loss_a: Treatment loss
    # loss_output: Output loss
    # loss_toxic: Toxicity loss
    
    # Setting model to evaluation mode
    model=model.eval()
    
    # Computing data interpolation
    if model.interpolation=='cubic':
        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables)
    elif model.interpolation=='linear':
        if rectilinear_index is None:
            test_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=covariables.shape[2]-1)
        else:
            test_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=rectilinear_index)
    
    # Computing predictions with the Neural CDE
    pred_output, pred_a,_ = model(test_coeffs, max_T=covariables.shape[1],static=static)
    
    # Initializing losses
    mseloss = torch.nn.MSELoss()

    if model.continuous: 
        if a_loss=='spearman':
            cost=torch.empty(size=[pred_a.shape[1]],device=model.device)
            for i in range(pred_a.shape[1]):
                loss=torch.empty(size=[1],device=model.device)
                loss[:]=np.nan
                for j in range(pred_a.shape[2]):
                    if j == 0 and len(validation_treatments[:,i+1,j:j+1][active_entries[:,i+1,:].bool()])>1:
                        loss = spearman(pred_a[:,i,j:j+1][active_entries[:,i+1,:].bool()], validation_treatments[:,i+1,j:j+1][active_entries[:,i+1,:].bool()],empty_var=empty_var)
                    elif len(validation_treatments[:,i+1,j:j+1][active_entries[:,i+1,:].bool()])>1:
                        loss=loss + spearman(pred_a[:,i,j:j+1][active_entries[:,i+1,:].bool()], validation_treatments[:,i+1,j:j+1][active_entries[:,i+1,:].bool()],empty_var=empty_var)
                cost[i] = loss
            loss_a=torch.nanmean(cost)

    else:
        treatment_loss = nn.CrossEntropyLoss()
        tl = nn.Softsign()
    
        # Computing the losses
        loss_a_compl = torch.empty([pred_a.shape[1],pred_a.shape[2]],device=model.device)
        for i in range(pred_a.shape[1]):
            for j in range(pred_a.shape[2]):
                loss_a_compl[i,j]=tl(treatment_loss(pred_a[:,i,j][active_entries[:,i+1,0].bool()], validation_treatments[:,i+1,j][active_entries[:,i+1,0].bool()]))
    
        loss_a = -torch.nanmean(loss_a_compl)
        
    if validation_output.shape[1]==active_entries.shape[1]:
        validation_output=validation_output[:,:-1,:]
    if validation_toxic.shape[1]==active_entries.shape[1]:
       validation_toxic=validation_toxic[:,:-1,:]
    
    loss_output=mseloss(pred_output[:,:,0:1][(active_entries[:,1:,:].bool()) & (~validation_output.isnan())], validation_output[(active_entries[:,1:,:].bool()) & (~validation_output.isnan())])
    loss_toxic_tensor = torch.empty(size=[validation_toxic.shape[2]],device=model.device)
    for j in range(validation_toxic.shape[2]):
        loss_toxic_tensor[j] = mseloss((pred_output[:,:,j+1:j+2][(active_entries[:,1:,:].bool()) & (~validation_toxic[:,:,j:j+1].isnan())]), validation_toxic[:,:,j:j+1][(active_entries[:,1:,:].bool()) & (~validation_toxic[:,:,j:j+1].isnan())])
    loss_toxic = torch.nanmean(loss_toxic_tensor)
        
    return pred_output, pred_a, loss_a, loss_output, loss_toxic

def predict_opt(model, data, a_loss='spearman',offset=5, a=None):
    # function using for treatment optimization with the synthetic cancer dataset
    
    model=model.eval()
    
    data2=copy.deepcopy(data)
    covariables_x = torch.from_numpy(np.concatenate([data2["current_covariates"][0:1,:,:],data2["current_treatments"][0:1,:,:]],axis=2))
    covariables_x = torch.tensor(covariables_x).float().to(model.device)
    time_covariates = torch.from_numpy(data2["time_covariates"][0:1,:,:])

    time_covariates = torch.tensor(time_covariates).float().to(model.device)
    covariables_x = torch.cat([covariables_x,time_covariates],dim=2)
    if offset>0:
        covariables_x=covariables_x[:,:offset+1,:]
    else:
        covariables_x=covariables_x[:,:2,:]
        covariables_x[:,-1,:-1] = torch.nan
    
    #if offset>1:
    if model.interpolation=='cubic':
        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables_x)
    elif model.interpolation=='linear':
        test_coeffs = torchcde.linear_interpolation_coeffs(covariables_x, rectilinear=covariables_x.shape[2]-1)
    #coeffs=test_coeffs
    pred_output_val, pred_a_val, _ = model(test_coeffs, max_T=covariables_x.shape[1])
    
    if offset>0:
        return pred_output_val
    else:
        return pred_output_val[:,0:1,:]

def prediction_measures(model, data_map, validation_output, validation_toxic, validation_treatments, covariables, active_entries, static,unscaled=False, rectilinear_index=None, step=None, variables_std=None,variables_mean=None,variables=None):
    # Function to compute prediction measures: mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape
    
    # Inputs: 
        
    # model: Neural CDE model
    # validation_output: Tensor of output/treatment success, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # validation_toxic: Tensor of the side effects, size: number of patients x timepoints (in hours) x number of side effects
    # validation_treatments: Tensor of the treatments, size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables, size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    # static: Tensor of static variables: number of patients x number of variables, default: None
    # rectilinear_index: Time index of covariables tensor, default: None
    # step: Corresponds to number of timestamps aggregated for computing the measures, default: None
    # unscaled: Inficator, whether variables should be normalized or not, default False
    # variables_std: list of the standard deviations of the variables, default None
    # variables_mean: list of the means of the variables, default None
    # variables: list of all variables
    
    # Outputs: 
    # mse: Mean Square Errors, size: number of patients x timepoints x number of variables
    # rmse: Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_sd: Normalized (standard deviation) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_mean: Normalized (mean) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_iqr: Normalized (inter-quartile range) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # mape: Mean Absolute Percentage Error, size: number of patients x timepoints x number of variables
    # mae: Mean Absolute Error, size: number of patients x timepoints x number of variables
    # wape: Weighted Absolute Percentage Error, size: number of patients x timepoints x number of variables
        
    mseloss = torch.nn.MSELoss()
    
    X=validation_output
    X_toxic=validation_toxic
    
    #Computing predictions
    pred_output_val, pred_a_val, loss_a_val, loss_output_val, loss_toxic_val = predict(model, validation_output=X, validation_toxic=X_toxic, validation_treatments=validation_treatments, covariables=covariables, active_entries=active_entries, static=static, rectilinear_index=rectilinear_index)

    if unscaled:
        if data_map is not None:
            pred_output_val[:,:,0] = pred_output_val[:,:,0]*data_map["output_stds"]+data_map["output_means"]
            pred_output_val[:,:,1] = pred_output_val[:,:,1]*data_map["output_toxicity_stds"]+data_map["output_toxicity_means"]
        else:
            pred_output_val[:,:,0] = pred_output_val[:,:,0]*variables_std[0]+variables_mean[0]
            pred_output_val[:,:,1] = pred_output_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
            pred_output_val[:,:,2] = pred_output_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
            pred_output_val[:,:,3] = pred_output_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    
    if X.shape[1]==active_entries.shape[1]:
        X=X[:,:-1,:]
    if X_toxic.shape[1]==active_entries.shape[1]:
        X_toxic=X_toxic[:,:-1,:]
    
    # Initializing mse loss
    mse = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    mse[:] = np.nan
    
    # Computing MSE
    if step is None:
        for i in range(mse.shape[0]):
            mse[i,0] = mseloss(pred_output_val[:,i,0:1][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())], X[:,i,:][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())])
            for j in range(mse.shape[1]-1):
                mse[i,j+1] = mseloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,i+1,:].bool()) & (~X_toxic[:,i,j:j+1].isnan())], X_toxic[:,i,j:j+1][(active_entries[:,i+1,0:1].bool()) & (~X_toxic[:,i,j:j+1].isnan())])
    else:
        for i in range(0,mse.shape[0],step):
            mse[i,0] = mseloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())], X[:,i:i+step,:][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())])
            for j in range(mse.shape[1]-1):
                mse[i,j+1] = mseloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,i+1:i+1+step,:].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())], X_toxic[:,i:i+step,j:j+1][(active_entries[:,i+1:i+1+step,0:1].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())])
        
    rmse=torch.sqrt(mse)
    
    # other potential measures
    X_std=X.clone()
    X_std[:,:,0][(~active_entries[:,1:,0].bool()) & (X[:,:,0].isnan())] = np.nan
    X_toxic_std=X_toxic.clone()
    
    nrmse_sd = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    nrmse_sd[:] = np.nan
    
    nrmse_mean = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    nrmse_mean[:] = np.nan
    
    nrmse_iqr = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    nrmse_iqr[:] = np.nan
    
    for j in range(X_toxic.shape[2]):
        X_toxic_std[:,:,j][(~active_entries[:,1:,0].bool()) & (X_toxic[:,:,j].isnan())] = np.nan
    
    if step is None:
        dat_sd = torch.from_numpy(np.nanstd(torch.cat([X_std[:,:,:],X_toxic_std[:,:,:]],dim=-1).cpu(),axis=0)).to(model.device)
        nrmse_sd = rmse/dat_sd
        
        dat_mean = torch.from_numpy(np.nanmean(torch.cat([X_std[:,:,:],X_toxic_std[:,:,:]],dim=-1).cpu(),axis=0)).to(model.device)
        nrmse_mean = rmse/dat_mean
        
        dat_iqr = torch.from_numpy(np.nanquantile(torch.cat([X_std[:,:,:],X_toxic_std[:,:,:]],dim=-1).cpu(),q=0.75,axis=0)).to(model.device) - torch.from_numpy(np.nanquantile(torch.cat([X_std[:,:,:],X_toxic_std[:,:,:]],dim=-1).cpu(),q=0.25,axis=0)).to(model.device)
        nrmse_iqr=rmse/dat_iqr
    else:
        for i in range(0,mse.shape[0],step):
            dat_sd = torch.from_numpy(np.nanstd(torch.cat([X_std[:,i:i+step,:],X_toxic_std[:,i:i+step,:]],dim=-1).cpu(),axis=(0,1))).to(model.device)
            dat_mean = torch.from_numpy(np.nanmean(torch.cat([X_std[:,i:i+step,:],X_toxic_std[:,i:i+step,:]],dim=-1).cpu(),axis=(0,1))).to(model.device)
            dat_iqr = torch.from_numpy(np.nanquantile(torch.cat([X_std[:,i:i+step,:],X_toxic_std[:,i:i+step,:]],dim=-1).cpu(),q=0.75,axis=(0,1))).to(model.device) - torch.from_numpy(np.nanquantile(torch.cat([X_std[:,i:i+step,:],X_toxic_std[:,i:i+step,:]],dim=-1).cpu(),q=0.25,axis=(0,1))).to(model.device)
       
            nrmse_sd[i] = rmse[i]/dat_sd
            
            nrmse_mean[i] = rmse[i]/dat_mean
            
            nrmse_iqr[i]=rmse[i]/dat_iqr
    
    mapeloss=torchmetrics.MeanAbsolutePercentageError().to(model.device)
    mape = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    mape[:]=np.nan
    
    if step is None:    
        for i in range(mape.shape[0]):
            mape[i,0] = mapeloss(pred_output_val[:,i,0:1][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())], X[:,i,:][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())])
            for j in range(mape.shape[1]-1):
                mape[i,j+1] = mapeloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,i+1,:].bool()) & (~X_toxic[:,i,j:j+1].isnan())], X_toxic[:,i,j:j+1][(active_entries[:,i+1,0:1].bool()) & (~X_toxic[:,i,j:j+1].isnan())])
    else:
        for i in range(0,mape.shape[0],step):
            mape[i,0] = mapeloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())], X[:,i:i+step,:][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())])
            for j in range(mape.shape[1]-1):
                mape[i,j+1] = mapeloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,i+1:i+1+step,:].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())], X_toxic[:,i:i+step,j:j+1][(active_entries[:,i+1:i+1+step,0:1].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())])
        

    wapeloss=torchmetrics.WeightedMeanAbsolutePercentageError().to(model.device)
    wape = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    wape[:]=np.nan
    
    if step is None:
        for i in range(mape.shape[0]):
            wape[i,0] = wapeloss(pred_output_val[:,i,0:1][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())], X[:,i,:][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())])
            for j in range(wape.shape[1]-1):
                wape[i,j+1] = wapeloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,i+1,:].bool()) & (~X_toxic[:,i,j:j+1].isnan())], X_toxic[:,i,j:j+1][(active_entries[:,i+1,0:1].bool()) & (~X_toxic[:,i,j:j+1].isnan())])
    else:
        for i in range(0,wape.shape[0],step):
            wape[i,0] = wapeloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())], X[:,i:i+step,:][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())])
            for j in range(wape.shape[1]-1):
                wape[i,j+1] = wapeloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,i+1:i+1+step,:].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())], X_toxic[:,i:i+step,j:j+1][(active_entries[:,i+1:i+1+step,0:1].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())])
        
    maeloss=torchmetrics.MeanAbsoluteError().to(model.device)
    mae = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    mae[:]=np.nan
    
    if step is None:
        for i in range(mape.shape[0]):
            mae[i,0] = maeloss(pred_output_val[:,i,0:1][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())], X[:,i,:][(active_entries[:,i+1,:].bool()) & (~X[:,i,:].isnan())])
            for j in range(mae.shape[1]-1):
                mae[i,j+1] = maeloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,i+1,:].bool()) & (~X_toxic[:,i,j:j+1].isnan())], X_toxic[:,i,j:j+1][(active_entries[:,i+1,0:1].bool()) & (~X_toxic[:,i,j:j+1].isnan())])    
    else:
        for i in range(0,mae.shape[0],step):
            mae[i,0] = maeloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())], X[:,i:i+step,:][(active_entries[:,i+1:i+1+step,:].bool()) & (~X[:,i:i+step,:].isnan())])
            for j in range(mae.shape[1]-1):
                mae[i,j+1] = maeloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,i+1:i+1+step,:].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())], X_toxic[:,i:i+step,j:j+1][(active_entries[:,i+1:i+1+step,0:1].bool()) & (~X_toxic[:,i:i+step,j:j+1].isnan())])
        
    return mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape

def train_dec_offset(model,model_decoder,train_output, train_toxic, train_treatments, covariables, time_covariates, active_entries, validation_output, validation_toxic, validation_treatments, covariables_val, validation_time_covariates, active_entries_val, static,static_val, offset=0, max_horizon=50, lr=0.001, batch_size=500, patience=10, delta=0.0001, max_epochs=1000, weight_loss=True, rectilinear_index=None, offset_train_list=None, offset_val_list=None, early_stop_path=None, dec_expand=False, mu=None,med_dec=True,med_dec_start=False,a_loss='spearman',empty_var=False):
    
    # Inputs:
    
    # model: Encoder object of the class Neural CDE
    # model_decoder: Decoder object of the class Neural CDE
    # train_output: Tensor of output/treatment success (training data), size: number of patients x timepoints (in hours) x 1 (dim of output)
    # train_toxic: Tensor of the side effects (training data), size: number of patients x timepoints (in hours) x number of side effects
    # train_treatments: Tensor of the treatments (training data), size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables (training data), size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    
    # validation_output: Tensor of output/treatment success (validation data), size: number of patients x timepoints (in hours) x 1 (dim of output)
    # validation_toxic: Tensor of the side effects (validation data), size: number of patients x timepoints (in hours) x number of side effects
    # validation_treatments: Tensor of the treatments (validation data), size: number of patients x timepoints (in hours), x number of treatments
    # covariables_val: Tensor of the covariables (validation data), size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries_val: Boolean Tensor indicating, whether patients is at ICU or discharged (validation data), size: number of patients x timepoints (in hours) x 1
    
    # static: Tensor of static variables (training data): batch x number of variables, default: None
    # static_val: Tensor of static variables (validation data): batch x number of variables, default: None
    
    # offset: Minimum number of observed time indices for Prediction with the Decoder, default: 0
    # max_horizon: Maximum number of (trained!) prediction horizon, default: 50
    # time_covariates: tensor of timepoints (of training data), size: batch x timepoints (in hours) x 1
    # validation_time_covariates: tensor of timepoints (of validation data), size: batch x timepoints (in hours) x 1
    
    # offset_train_list: List of all offsets used for training (subset of all offsets due to long training duration), default: None
    # offset_train_list: List of all offsets used for validation, default: None
    
    # dec_expand Indicating whether initialization of decoder is expanded (at least with static data) or not default: False
    # med_dec: Indicating, whether Treatments are used as control or not default: True
    # med_dec_start: Indicating, whether initialization of decoder is expanded by the last measured treatment default: False
    
    # Outputs:
    # loss_val: Validation loss (necessary for hyperparameteroptimization)
    
    # Setting model to evaluation mode
    model=model.eval()
    optimizer = torch.optim.Adam(model_decoder.parameters(),lr=lr)

    early_stopping = EarlyStopping(patience=patience, delta=delta, path=early_stop_path)
    
    # Computing interpolation of the data
    if model.interpolation=='cubic':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables)
        validation_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables_val)
    elif model.interpolation=='linear':
        if rectilinear_index is None:
            train_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=covariables.shape[2]-1)
            validation_coeffs = torchcde.linear_interpolation_coeffs(covariables_val, rectilinear=covariables_val.shape[2]-1)
        else:
            train_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=rectilinear_index)
            validation_coeffs = torchcde.linear_interpolation_coeffs(covariables_val, rectilinear=rectilinear_index)
    
    # Initializing dataloader and loss functions
    if static is not None:
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_output, active_entries, train_toxic, train_treatments, time_covariates, train_treatments, covariables, static)
    else:
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_output, active_entries, train_toxic, train_treatments, time_covariates, train_treatments, covariables)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    # Setting model (Encoder) to evaluation mode
    model=model.eval()
    
    mseloss = torch.nn.MSELoss()
    
    if not model.continuous:
        treatment_loss = nn.CrossEntropyLoss()
        tl = nn.Softsign()
    
    offset_pre=offset
    max_horizon_pre=max_horizon
    
    for epoch in range(max_epochs):
        for batch in train_dataloader:
            # Initializing loss functions
            loss_output_batch = torch.zeros(size=[1],device=model.device)
            loss_toxic_batch = torch.zeros(size=[1],device=model.device)
            loss_a_batch = torch.zeros(size=[1],device=model.device)
            offset=offset_pre
            max_horizon=max_horizon_pre
            
            #Computing batches
            if static is not None:
                batch_coeffs, train_output_b, active_entries_b, train_toxic_b, train_treat_b, time_covariates_b, train_treatments_b, covariables_b, static_b = batch
            else:
                batch_coeffs, train_output_b, active_entries_b, train_toxic_b, train_treat_b, time_covariates_b, train_treatments_b, covariables_b = batch
                static_b=None
            for i in range(max_horizon_pre-2):
                #Checking whether model is trained for i'th offset
                if offset_train_list is None or i in offset_train_list:
                    # Computing predictions 
                    pred_output, pred_a,zt = model(batch_coeffs, max_T=offset+1,static=static_b)
                    
                    #Computing "control of decoder"
                    if med_dec:
                        train_covariables_dec = train_treatments_b[:,offset+1:offset+max_horizon+1,:]
                        train_covariables_dec = torch.cat([train_covariables_dec,time_covariates_b[:,offset+1:offset+max_horizon+1,:]],axis=-1)
                    else:
                        train_covariables_dec = time_covariates_b[:,offset+1:offset+max_horizon+1,:]
                    zt_cov = zt[:,offset,:].detach() 

                    zt_cov = zt_cov[:,None,:]
                    
                    # Expanding initialization of decoder
                    if dec_expand:
                            zt_cov = torch.cat([zt_cov,static],axis=-1)
                    
                    # Computing interpolation of training data of decoder
                    if model_decoder.interpolation=='cubic':
                        train_coeffs_dec = torchcde.hermite_cubic_coefficients_with_backward_differences(train_covariables_dec)
                    elif model.interpolation=='linear':
                        train_coeffs_dec = torchcde.linear_interpolation_coeffs(train_covariables_dec, rectilinear=train_covariables_dec.shape[2]-1)
                    
                    # Setting Decoder to training mode
                    model_decoder=model_decoder.train()
                    
                    # Computing predictions of the Decoder
                    pred_output, pred_a, _ = model_decoder(train_coeffs_dec, max_T=max_horizon,z0=zt_cov[:,0,:])
                    
                    if model_decoder.continuous:
                        if a_loss=='spearman':
                            cost=torch.empty(size=[pred_a.shape[1]],device=model.device)
                            for i in range(pred_a.shape[1]):
                                loss=torch.empty(size=[1],device=model.device)
                                loss[:]=np.nan
                                for j in range(pred_a.shape[2]):
                                    if j == 0 and len(train_treat_b[:,offset+2+i,j:j+1][active_entries_b[:,offset+2+i,:].bool()])>1:
                                        loss=spearman(pred_a[:,i,j:j+1][active_entries_b[:,offset+2+i,:].bool()], train_treat_b[:,offset+2+i,j:j+1][active_entries_b[:,offset+2+i,:].bool()],empty_var=empty_var)
                                    elif train_treat_b[:,offset+2+i,j:j+1].shape[1]>0 and len(train_treat_b[:,offset+2+i,j:j+1][active_entries_b[:,offset+2+i,:].bool()])>1:
                                        loss=loss + spearman(pred_a[:,i,j:j+1][active_entries_b[:,offset+2+i,:].bool()], train_treat_b[:,offset+2+i,j:j+1][active_entries_b[:,offset+2+i,:].bool()],empty_var=empty_var)
                                cost[i] = loss
                            loss_a=torch.nanmean(cost)
                    
                    else:
                        # Initializing and computing losses
                        loss_a_compl = torch.empty([pred_a.shape[1],pred_a.shape[2]],device=model.device)
                        loss_a_compl[:] = np.nan
                        for i in range(pred_a.shape[1]):
                            for j in range(pred_a.shape[2]):
                                loss_a_compl[i,j]=tl(treatment_loss(pred_a[:,i,j][active_entries_b[:,offset+2+i,0].bool()], train_treat_b[:,offset+2+i,j][active_entries_b[:,offset+2+i,0].bool()]))
                    
                        loss_a = -torch.nanmean(loss_a_compl)
                    
                    loss_output=mseloss(pred_output[:,:,0:1][(active_entries_b[:,offset+2:offset+max_horizon+2,:].bool()) & (~train_output_b[:,offset+1:offset+max_horizon+1,:].isnan())], train_output_b[:,offset+1:offset+max_horizon+1,:][(active_entries_b[:,offset+2:offset+max_horizon+2,:].bool())& (~train_output_b[:,offset+1:offset+max_horizon+1,:].isnan())])
                    
                    loss_toxic_tensor = torch.empty(size=[train_toxic_b.shape[2]],device=model.device)
                    loss_toxic_tensor[:] = np.nan
                    for j in range(train_toxic_b.shape[2]):
                        loss_toxic_tensor[j] = mseloss((pred_output[:,:,j+1:j+2][(active_entries_b[:,offset+2:offset+max_horizon+2,:].bool()) & (~train_toxic_b[:,offset+1:offset+max_horizon+1,j:j+1].isnan())]), train_toxic_b[:,offset+1:offset+max_horizon+1,j:j+1][(active_entries_b[:,offset+2:offset+max_horizon+2,:].bool()) & (~train_toxic_b[:,offset+1:offset+max_horizon+1,j:j+1].isnan())])
                    loss_toxic = torch.nanmean(loss_toxic_tensor)
                        
                    if ~loss_output.isnan().any():
                        loss_output_batch = loss_output_batch + loss_output
                    if ~loss_toxic.isnan().any():
                        loss_toxic_batch = loss_toxic_batch + loss_toxic
                    if ~loss_a.isnan().any():
                        loss_a_batch = loss_a_batch + loss_a
                            
                # Setting offset +1 and max_horizon -1 (here: Max_horizon is not maximum prediction horizon, but prediction horizon wrt offset!!)
                offset=offset+1
                max_horizon=max_horizon-1
            
            # Computing mu
            if mu is None:
                if weight_loss:
                    mu = 0
                    mu = model.output_channels/model.treatment_options
                else:
                    mu=1
            
            #Computing loss and optimizing
            if loss_output_batch != 0 and loss_toxic_batch != 0 and loss_a_batch != 0:
                loss_dec= loss_output_batch + loss_toxic_batch + mu * loss_a_batch
            else:
                loss_dec=torch.Tensor([np.nan])
                
            loss_dec.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss_output_batch)
        print(loss_toxic_batch)
        print(loss_a_batch)
        print('Epoch: {}   Training loss: {}'.format(epoch, loss_dec.item()))
        
        # Setting Encoder and Decoder to evaluation mode
        model=model.eval()
        model_decoder=model_decoder.eval()
        
        offset=offset_pre
        max_horizon=max_horizon_pre
        
        for i in range(max_horizon_pre-2):
            
            # Initializing loss functions
            loss_output_val = torch.zeros(size=[1],device=model.device)
            loss_toxic_val = torch.zeros(size=[1],device=model.device)
            loss_a_val = torch.zeros(size=[1],device=model.device)
            if offset_val_list is None or i in offset_val_list:
                pred_output_val, pred_a_val,zt = model(validation_coeffs, max_T=offset+1,static=static_val)
                
                if med_dec:
                    val_covariables_dec = validation_treatments[:,offset+1:offset+max_horizon+1,:]
                    val_covariables_dec = torch.cat([val_covariables_dec,validation_time_covariates[:,offset+1:offset+max_horizon+1,:]],axis=-1)
                else:
                    val_covariables_dec = validation_time_covariates[:,offset+1:offset+max_horizon+1,:]
                zt_val = zt[:,offset,:].detach() 
                    
                zt_val = zt_val[:,None,:]           
                if dec_expand:
                    zt_val = torch.cat([zt_val,static_val],axis=-1)
                
                if model_decoder.interpolation=='cubic':
                    validation_coeffs_dec = torchcde.hermite_cubic_coefficients_with_backward_differences(val_covariables_dec)
                elif model_decoder.interpolation=='linear':
                    validation_coeffs_dec = torchcde.linear_interpolation_coeffs(val_covariables_dec, rectilinear=val_covariables_dec.shape[2]-1)

                pred_output_val, pred_a_val, _ = model_decoder(validation_coeffs_dec,max_T=max_horizon,z0=zt_val[:,0,:])
                
                if model_decoder.continuous:
                    if a_loss=='spearman':
                        cost=torch.empty(size=[pred_a_val.shape[1]],device=model.device)
                        for i in range(pred_a_val.shape[1]):
                            loss=torch.empty(size=[1],device=model.device)
                            loss[:]=np.nan
                            for j in range(pred_a_val.shape[2]):
                                if j == 0 and len(validation_treatments[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()])>1:
                                    loss=spearman(pred_a_val[:,i,j:j+1][active_entries_val[:,offset+2+i,:].bool()], validation_treatments[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()],empty_var=empty_var)
                                elif validation_treatments[:,offset+2+i,j:j+1].shape[1]>0 and len(validation_treatments[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()])>1:
                                    loss=loss + spearman(pred_a_val[:,i,j:j+1][active_entries_val[:,offset+2+i,:].bool()], validation_treatments[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()],empty_var=empty_var)
                            cost[i] = loss
                        loss_a=torch.nanmean(cost)
                
                else:
                    loss_a_compl = torch.empty([pred_a_val.shape[1],pred_a_val.shape[2]],device=model.device)
                    loss_a_compl[:] = np.nan
                    for i in range(pred_a_val.shape[1]):
                        for j in range(pred_a.shape[2]):
                            loss_a_compl[i,j]=tl(treatment_loss(pred_a_val[:,i,j][active_entries_val[:,offset+2+i,0].bool()], validation_treatments[:,offset+2+i,j][active_entries_val[:,offset+2+i,0].bool()]))
                
                    loss_a = -torch.nanmean(loss_a_compl)
                
                loss_output= mseloss(pred_output_val[:,:,0:1][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~validation_output[:,offset+1:offset+max_horizon+1,:].isnan())], validation_output[:,offset+1:offset+max_horizon+1,:][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~validation_output[:,offset+1:offset+max_horizon+1,:].isnan())])
                loss_toxic_tensor = torch.empty(size=[validation_toxic.shape[2]],device=model.device)
                loss_toxic_tensor[:] = np.nan
                for j in range(validation_toxic.shape[2]):
                    loss_toxic_tensor[j] = mseloss((pred_output_val[:,:,j+1:j+2][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~validation_toxic[:,offset+1:offset+max_horizon+1,j:j+1].isnan())]), validation_toxic[:,offset+1:offset+max_horizon+1,j:j+1][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~validation_toxic[:,offset+1:offset+max_horizon+1,j:j+1].isnan())])
                loss_toxic = torch.nanmean(loss_toxic_tensor)
                    
                if ~loss_output.isnan().any():
                    loss_output_val = loss_output_val + loss_output
                if ~loss_toxic.isnan().any():
                    loss_toxic_val = loss_toxic_val + loss_toxic
                if ~loss_a.isnan().any():
                    loss_a_val = loss_a_val + loss_a
                    
            offset=offset+1
            max_horizon=max_horizon-1
        
        if mu is None:
            if weight_loss:
                mu = 0
                mu = model.output_channels/model.treatment_options
            else:
                mu=1
        
        # Computing validation loss
        print("validation")
        print(loss_a_val)
        print(loss_output_val)
        print(loss_toxic_val)
        if loss_output_val != 0 and loss_toxic_val != 0 and loss_a_val != 0:
            loss_val= loss_output_val + loss_toxic_val + mu * loss_a_val
        
        else:
            loss_val=torch.Tensor([np.nan])
        print('Epoch: {}   Validation loss: {}'.format(epoch, loss_val.item()))
        
        early_stopping(loss_val, model_decoder)
        
        if early_stopping.early_stop:
            break
        print("Early stopping phase initiated...")
    
    # Saving model
    
    torch.save(model_decoder.state_dict(), './final_model_decoder.pth')
    
    return loss_val

def predict_decoder_opt(model, model_decoder, data, offset=5, max_horizon=5, a_loss='spearman',a=None):
    # function using for treatment optimization 
    model=model.eval()
    model_decoder=model_decoder.eval()
    
    data2=copy.deepcopy(data)
    
    # a is treatment
    if a is not None:
        a2=copy.deepcopy(a)
        a2[:,:,0]=(a2[:,:,0]-data2["input_means"][2])/data2["inputs_stds"][2]
        a2[:,:,1]=(a2[:,:,1]-data2["input_means"][3])/data2["inputs_stds"][3]
        data2["current_treatments"][0:1,:max_horizon,:] = a2
    
    covariables_x = torch.from_numpy(np.concatenate([data2["current_covariates"][0:1,:,:],data2["current_treatments"][0:1,:,:]],axis=2)).float().to(model.device)
    time_covariates = torch.from_numpy(data2["time_covariates"][0:1,:,:]).float().to(model.device)
    covariables_x = torch.cat([covariables_x,time_covariates[0:1,:,:]],dim=2)
    if offset>0:
        covariables_x=covariables_x[:,:offset+1,:]
    else:
        covariables_x=covariables_x[:,:2,:]
        covariables_x[:,-1,:-1] = torch.nan
    
    if model.interpolation=='cubic':
        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables_x)
    elif model.interpolation=='linear':
        test_coeffs = torchcde.linear_interpolation_coeffs(covariables_x, rectilinear=covariables_x.shape[2]-1)

    X_enc, pred_a_val_enc, zt = model(test_coeffs, max_T=offset+1)
    
    zt_val = zt[:,offset,:].detach() 
    zt_val = zt_val[:,None,:]

    current_treatment_val=torch.from_numpy(data2["current_treatments"][0:1,offset+1:max_horizon,:]).float().to(model.device)
    if max_horizon-(offset+1)==1:
        val_covariables_dec = torch.cat([current_treatment_val,time_covariates[:,offset+1:max_horizon,:]],axis=-1)
        new_t = torch.cat([current_treatment_val,time_covariates[:,offset+1:max_horizon,:]],axis=-1)
        new_t[:,-1,:-1] = torch.nan
        #Be careful by introducing the time, it is important to use the correct spacing
        new_t[:,-1,-1] = new_t[:,-1,-1]+time_covariates[0,1,0]
        val_covariables_dec=torch.cat([val_covariables_dec,new_t],axis=1)
        
    else:
        val_covariables_dec = torch.cat([current_treatment_val,time_covariates[:,offset+1:max_horizon,:]],axis=-1)
        
    if model_decoder.interpolation=='cubic':
        validation_coeffs_dec = torchcde.hermite_cubic_coefficients_with_backward_differences(val_covariables_dec)
    elif model_decoder.interpolation=='linear':
        validation_coeffs_dec = torchcde.linear_interpolation_coeffs(val_covariables_dec, rectilinear=val_covariables_dec.shape[2]-1)
    #max_T to check
    if max_horizon-(offset+1)>1:
        X, pred_a_val, _ = model_decoder(validation_coeffs_dec, max_T=max_horizon-(offset+1),z0=zt_val[:,0,:])
    else:
        X, pred_a_val, _ = model_decoder(validation_coeffs_dec, max_T=2,z0=zt_val[:,0,:])
        X = X[:,0:1,:]
    #aim: minimizing at the last timepoint, but it is important the consider all predictions due to side-effects
    X = torch.cat([X_enc,X],dim=1)
    return X

def predict_decoder(model, model_decoder, validation_output, validation_toxic, validation_treatments, covariables, time_covariates, active_entries, static, offset=5, max_horizon=5, rectilinear_index=None, dec_expand=False,med_dec=True,med_dec_start=False,a_loss='spearman',empty_var=False):
    
    # Inputs:
    
    # model: Encoder object of the class Neural CDE
    # model_decoder: Decoder object of the class Neural CDE
    # train_output: Tensor of output/treatment success (training data), size: number of patients x timepoints (in hours) x 1 (dim of output)
    # train_toxic: Tensor of the side effects (training data), size: number of patients x timepoints (in hours) x number of side effects
    # train_treatments: Tensor of the treatments (training data), size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables (training data), size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    # static: Tensor of static variables (training data): batch x number of variables, default: None
    
    # offset: Minimum number of observed time indices for Prediction with the Decoder, default: 0
    # max_horizon: Maximum number of (trained!) prediction horizon, default: 50
    # time_covariates: tensor of timepoints (of training data), size: batch x timepoints (in hours) x 1

    # dec_expand Indicating whether initialization of decoder is expanded (at least with static data) or not default: False
    # sofa_expand: Indicating whether initialization of decoder is expanded by last measured sofa_score default: False
    # med_dec: Indicating, whether Treatments are used as control or not default: True
    # med_dec_start: Indicating, whether initialization of decoder is expanded by the last measured treatment default: False
    
    # output:
    # pred_output: Tensor of the predicted outcomes, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # pred_a: Tensor of the predicted side-effects associated laboratory variables size: number of patients x timepoints (in hours) x number of side effects
    # loss_a: Treatment loss
    # loss_output: Output loss
    # loss_toxic: Toxicity loss
    
    
    model=model.eval()
    model_decoder=model_decoder.eval()
    
    X=validation_output
    X_toxic=validation_toxic
    current_treatment_val=validation_treatments
    active_entries_val=active_entries
    
    if offset>0:
        covariables=covariables[:,:offset+1,:]
    else:
        covariables=covariables[:,:2,:]
        covariables[:,-1,:-1] = torch.nan
    
    if model.interpolation=='cubic':
        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(covariables)
    elif model.interpolation=='linear':
        if rectilinear_index is None:
            test_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=covariables.shape[2]-1)
        else:
            test_coeffs = torchcde.linear_interpolation_coeffs(covariables, rectilinear=rectilinear_index)
    
    mseloss = torch.nn.MSELoss()
    
    if not model.continuous:
        treatment_loss = nn.CrossEntropyLoss()
        tl = nn.Softsign()
    
    pred_output, pred_a,zt = model(test_coeffs, max_T=offset+1,static=static)

    if max_horizon==1:
        if med_dec == True:
            val_covariables_dec = torch.cat([current_treatment_val[:,offset+1:offset+max_horizon+1,:],time_covariates[:,offset+1:offset+max_horizon+1,:]],axis=-1)
        else:
            val_covariables_dec = time_covariates[:,offset+1:offset+max_horizon+1,:]
        new_t = torch.cat([current_treatment_val[:,offset+1:offset+max_horizon+1,:],time_covariates[:,offset+1:offset+max_horizon+1,:]],axis=-1)
        new_t[:,-1,:-1] = torch.nan
        #Be careful by introducing the time, it is important to use the correct spacing
        new_t[:,-1,-1] = new_t[:,-1,-1]+time_covariates[0,1,0]
    else:
        if med_dec==True:
            val_covariables_dec = current_treatment_val[:,offset+1:offset+max_horizon+1,:]
            val_covariables_dec = torch.cat([val_covariables_dec,time_covariates[:,offset+1:offset+max_horizon+1,:]],axis=-1)
        else:
            val_covariables_dec = time_covariates[:,offset+1:offset+max_horizon+1,:]
    zt_val = zt[:,offset,:].detach() 

    
    zt_val = zt_val[:,None,:]           
    if dec_expand:
        zt_val = torch.cat([zt_val,static],axis=-1)
    
    if model_decoder.interpolation=='cubic':
        validation_coeffs_dec = torchcde.hermite_cubic_coefficients_with_backward_differences(val_covariables_dec)
    elif model_decoder.interpolation=='linear':
        validation_coeffs_dec = torchcde.linear_interpolation_coeffs(val_covariables_dec, rectilinear=val_covariables_dec.shape[2]-1)
    

    pred_output_val, pred_a_val, _ = model_decoder(validation_coeffs_dec, max_T=max_horizon,z0=zt_val[:,0,:])

    if max_horizon==1:
        pred_output_val, pred_a_val, _ = model_decoder(validation_coeffs_dec, max_T=2,z0=zt_val[:,0,:])
        pred_output_val = pred_output_val[:,0:1,:]
    
    if model.continuous:
        if a_loss=='spearman':
            cost=torch.empty(size=[pred_a_val.shape[1]],device=model.device)
            for i in range(pred_a_val.shape[1]):
                loss=torch.empty(size=[1],device=model.device)
                loss[:]=np.nan
                for j in range(current_treatment_val.shape[2]):
                    if j == 0 and len(current_treatment_val[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()])>1:
                        loss=spearman(pred_a_val[:,i,j:j+1][active_entries_val[:,offset+2+i,:].bool()], current_treatment_val[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()],empty_var=empty_var)
                    elif len(current_treatment_val[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()])>1:
                        loss=loss + spearman(pred_a_val[:,i,j:j+1][active_entries_val[:,offset+2+i,:].bool()], current_treatment_val[:,offset+2+i,j:j+1][active_entries_val[:,offset+2+i,:].bool()],empty_var=empty_var)
                cost[i] = loss
            loss_a=torch.nanmean(cost)
    else:
        loss_a_compl = torch.empty([pred_a_val.shape[1],pred_a_val.shape[2]],device=model.device)
        loss_a_compl[:] = np.nan
        for i in range(pred_a_val.shape[1]):
            for j in range(pred_a_val.shape[2]):
                loss_a_compl[i,j]=tl(treatment_loss(pred_a_val[:,i,j][active_entries_val[:,offset+2+i,0].bool()], current_treatment_val[:,offset+2+i,j][active_entries_val[:,offset+2+i,0].bool()]))
    
        loss_a = -torch.nanmean(loss_a_compl)

    loss_output_val=mseloss(pred_output_val[:,:,0:1][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~X[:,offset+1:offset+max_horizon+1,:].isnan())], X[:,offset+1:offset+max_horizon+1,:][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool())& (~X[:,offset+1:offset+max_horizon+1,:].isnan())])
    
    loss_toxic_tensor_val = torch.empty(size=[X_toxic.shape[2]],device=model.device)
    loss_toxic_tensor_val[:]=np.nan
    for j in range(X_toxic.shape[2]):
        loss_toxic_tensor_val[j] = mseloss((pred_output_val[:,:,j+1:j+2][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~X_toxic[:,offset+1:offset+max_horizon+1,j:j+1].isnan())]), X_toxic[:,offset+1:offset+max_horizon+1,j:j+1][(active_entries_val[:,offset+2:offset+max_horizon+2,:].bool()) & (~X_toxic[:,offset+1:offset+max_horizon+1,j:j+1].isnan())])
    loss_toxic_val = torch.nanmean(loss_toxic_tensor_val)
        
        
    return pred_output_val, pred_a_val, loss_a, loss_output_val, loss_toxic_val


def prediction_measures_decoder(model, model_decoder, data_map, validation_output, validation_toxic, validation_treatments, covariables, time_covariates, active_entries, static=None, offset=5,max_horizon=5,unscaled=False, rectilinear_index=None, step=None, variables_std=None,variables_mean=None,variables=None, dec_expand=False, med_dec=True, med_dec_start=False):
    # Function to compute prediction performance metrics for the long-term prediction (decoder)
    
    # Inputs: 
        
    # model: Neural CDE model of the Encoder
    # model_decoder: Neural CDE model of the Decoder
    # validation_output: Tensor of output/treatment success, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # validation_toxic: Tensor of the side effects, size: number of patients x timepoints (in hours) x number of side effects
    # validation_treatments: Tensor of the treatments, size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables, size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    # static: Tensor of static variables: number of patients x number of variables, default: None
    # rectilinear_index: Time index of covariables tensor, default: None

    # step: Corresponds to number of timestamps aggregated for computing the measures, default: None
    # unscaled: Inficator, whether variables should be normalized or not, default False
    # variables_std: list of the standard deviations of the variables, default None
    # variables_mean: list of the means of the variables, default None
    # variables: list of all variables

    # dec_expand Indicating whether initialization of decoder is expanded (at least with static data) or not default: False
    # sofa_expand: Indicating whether initialization of decoder is expanded by last measured sofa_score default: False
    # med_dec: Indicating, whether Treatments are used as control or not default: False
    # med_dec_start: Indicating, whether initialization of decoder is expanded by the last measured treatment default: True    
    
    # Outputs: 
    # mse: Mean Square Errors, size: number of patients x timepoints x number of variables
    # rmse: Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_sd: Normalized (standard deviation) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_mean: Normalized (mean) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_iqr: Normalized (inter-quartile range) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # mape: Mean Absolute Percentage Error, size: number of patients x timepoints x number of variables
    # mae: Mean Absolute Error, size: number of patients x timepoints x number of variables
    # wape: Weighted Absolute Percentage Error, size: number of patients x timepoints x number of variables
        
    
    mseloss = torch.nn.MSELoss()

    X=validation_output
    X_toxic=validation_toxic

    pred_output_val, pred_a_val, loss_a_val, loss_output_val, loss_toxic_val = predict_decoder(model, model_decoder, offset=offset, max_horizon=max_horizon, validation_output=X, validation_toxic=X_toxic, validation_treatments=validation_treatments, covariables=covariables, time_covariates=time_covariates, active_entries=active_entries, static=static, rectilinear_index=rectilinear_index, dec_expand=dec_expand, med_dec=med_dec, med_dec_start=med_dec_start)

    if unscaled:
        if data_map is not None:
            X[:,:,0] = X[:,:,0]*data_map["output_stds"]+data_map["output_means"]
            X[:,:,1] = X[:,:,1]*data_map["output_toxicity_stds"]+data_map["output_toxicity_means"]
        else:
            X[:,:,0] = X[:,:,0]*variables_std[0]+variables_mean[0]
            X[:,:,1] = X[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
            X[:,:,2] = X[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
            X[:,:,3] = X[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
        
    if X.shape[1]==active_entries.shape[1]:
        X=X[:,:-1,:]
    if X_toxic.shape[1]==active_entries.shape[1]:
        X_toxic=X_toxic[:,:-1,:]
        
        
    mse = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    mse[:] = np.nan

    if step is None:
        for i in range(mse.shape[0]):
            mse[i,0] = mseloss(pred_output_val[:,i,0:1][(active_entries[:,offset+i+2,:].bool()) & (~X[:,offset+i+1,:].isnan())], X[:,offset+i+1,:][(active_entries[:,offset+2+i,:].bool()) & (~X[:,offset+i+1,:].isnan())])
            for j in range(mse.shape[1]-1):
                mse[i,j+1] = mseloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,offset+i+2,:].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())], X_toxic[:,offset+i+1,j:j+1][(active_entries[:,offset+2+i,0:1].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())])
    else:
        for i in range(0,mse.shape[0],step):
            mse[i,0] = mseloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())], X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())])
            for j in range(mse.shape[1]-1):
                mse[i,j+1] = mseloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())], X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0]):1].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())])
        
    rmse=torch.sqrt(mse)

    X_std=X.clone()
    X_std[:,:,0][(~active_entries[:,1:,0].bool()) & (X[:,:,0].isnan())] = np.nan
    X_toxic_std=X_toxic.clone()

    nrmse_sd = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    nrmse_sd[:] = np.nan

    nrmse_mean = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    nrmse_mean[:] = np.nan

    nrmse_iqr = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    nrmse_iqr[:] = np.nan

    for j in range(X_toxic.shape[2]):
        X_toxic_std[:,:,j][(~active_entries[:,1:,0].bool()) & (X_toxic[:,:,j].isnan())] = np.nan

    if step is None:
        dat_sd = torch.from_numpy(np.nanstd(torch.cat([X_std[:,offset+1:offset+max_horizon+1,:],X_toxic_std[:,offset+1:offset+max_horizon+1,:]],dim=-1).cpu(),axis=0)).to(model.device)
        nrmse_sd = rmse/dat_sd
        
        dat_mean = torch.from_numpy(np.nanmean(torch.cat([X_std[:,offset+1:offset+max_horizon+1,:],X_toxic_std[:,offset+1:offset+max_horizon+1,:]],dim=-1).cpu(),axis=0)).to(model.device)
        nrmse_mean = rmse/dat_mean
        
        dat_iqr = torch.from_numpy(np.nanquantile(torch.cat([X_std[:,offset+1:offset+max_horizon+1,:],X_toxic_std[:,offset+1:offset+max_horizon+1,:]],dim=-1).cpu(),q=0.75,axis=0)).to(model.device) - torch.from_numpy(np.nanquantile(torch.cat([X_std[:,offset+1:offset+max_horizon+1,:],X_toxic_std[:,offset+1:offset+max_horizon+1,:]],dim=-1).cpu(),q=0.25,axis=0)).to(model.device)
        nrmse_iqr=rmse/dat_iqr
    else:
        for i in range(0,mse.shape[0],step):
            dat_sd = torch.from_numpy(np.nanstd(torch.cat([X_std[:,offset+1+i:offset+1+i+step,:],X_toxic_std[:,offset+1+i:offset+1+i+step,:]],dim=-1).cpu(),axis=(0,1))).to(model.device)
            dat_mean = torch.from_numpy(np.nanmean(torch.cat([X_std[:,offset+1+i:offset+1+i+step,:],X_toxic_std[:,offset+1+i:offset+1+i+step,:]],dim=-1).cpu(),axis=(0,1))).to(model.device)
            dat_iqr = torch.from_numpy(np.nanquantile(torch.cat([X_std[:,offset+1+i:offset+1+i+step,:],X_toxic_std[:,offset+1+i:offset+1+i+step,:]],dim=-1).cpu(),q=0.75,axis=(0,1))).to(model.device) - torch.from_numpy(np.nanquantile(torch.cat([X_std[:,offset+1+i:offset+1+i+step,:],X_toxic_std[:,offset+1+i:offset+1+i+step,:]],dim=-1).cpu(),q=0.25,axis=(0,1))).to(model.device)
       
            nrmse_sd[i] = rmse[i]/dat_sd
            
            nrmse_mean[i] = rmse[i]/dat_mean
            
            nrmse_iqr[i]=rmse[i]/dat_iqr

    mapeloss=torchmetrics.MeanAbsolutePercentageError().to(model.device)
    mape = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    mape[:] = np.nan

    if step is None:
        for i in range(mape.shape[0]):
            mape[i,0] = mapeloss(pred_output_val[:,i,0:1][(active_entries[:,offset+i+2,:].bool()) & (~X[:,offset+i+1,:].isnan())], X[:,offset+i+1,:][(active_entries[:,offset+2+i,:].bool()) & (~X[:,offset+i+1,:].isnan())])
            for j in range(mape.shape[1]-1):
                mape[i,j+1] = mapeloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,offset+i+2,:].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())], X_toxic[:,offset+i+1,j:j+1][(active_entries[:,offset+2+i,0:1].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())])
    else:
        for i in range(0,mape.shape[0],step):
            mape[i,0] = mapeloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())], X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0])].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())])
            for j in range(mape.shape[1]-1):
                mape[i,j+1] = mapeloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())], X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0]):1].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())])
        

    wapeloss=torchmetrics.WeightedMeanAbsolutePercentageError().to(model.device)
    wape = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    wape[:] = np.nan

    if step is None:
        for i in range(wape.shape[0]):
            wape[i,0] = wapeloss(pred_output_val[:,i,0:1][(active_entries[:,offset+i+2,:].bool()) & (~X[:,offset+i+1,:].isnan())], X[:,offset+i+1,:][(active_entries[:,offset+2+i,:].bool()) & (~X[:,offset+i+1,:].isnan())])
            for j in range(wape.shape[1]-1):
                wape[i,j+1] = wapeloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,offset+i+2,:].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())], X_toxic[:,offset+i+1,j:j+1][(active_entries[:,offset+2+i,0:1].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())])
    else:
        for i in range(0,wape.shape[0],step):
            wape[i,0] = wapeloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())], X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0])].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())])
            for j in range(wape.shape[1]-1):
                wape[i,j+1] = wapeloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())], X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0]):1].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())])
        

    maeloss=torchmetrics.MeanAbsoluteError().to(model.device)
    mae = torch.empty(size=pred_output_val.shape[1:],device=model.device)
    mae[:] = np.nan

    if step is None:
        for i in range(mae.shape[0]):
            mae[i,0] = maeloss(pred_output_val[:,i,0:1][(active_entries[:,offset+i+2,:].bool()) & (~X[:,offset+i+1,:].isnan())], X[:,offset+i+1,:][(active_entries[:,offset+2+i,:].bool()) & (~X[:,offset+i+1,:].isnan())])
            for j in range(mae.shape[1]-1):
                mae[i,j+1] = maeloss(pred_output_val[:,i,j+1:j+2][(active_entries[:,offset+i+2,:].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())], X_toxic[:,offset+i+1,j:j+1][(active_entries[:,offset+2+i,0:1].bool()) & (~X_toxic[:,offset+i+1,j:j+1].isnan())])
    else:
        for i in range(0,mae.shape[0],step):
            mae[i,0] = maeloss(pred_output_val[:,i:i+step,0:1][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())], X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0])].bool()) & (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),:].isnan())])
            for j in range(mae.shape[1]-1):
                mae[i,j+1] = maeloss(pred_output_val[:,i:i+step,j+1:j+2][(active_entries[:,offset+i+2:min(offset+2+i+step,offset+2+mse.shape[0]),:].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())], X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1][(active_entries[:,offset+2+i:min(offset+2+i+step,offset+2+mse.shape[0]):1].bool()) & (~X_toxic[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())])
        
    return mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape

def heatmap_pred_dec(model, model_decoder, data_map, validation_output, validation_toxic, validation_treatments, covariables, time_covariates, active_entries, static=None, index=0, offset=0, max_horizon=9,loss='rmse', rectilinear_index=None, step=None, unscaled=False, variables_std=None,variables_mean=None,variables=None,title=str(),save_link=None,save_map=None,load_map=None,vmin=None,vmax=None,colorbar=True,diagmultistep=False, dec_expand=False, med_dec=True, med_dec_start=False, invert=False):
    # Function to create a heatmap based on the prediction measures of the Decoder
    
    # model: Neural CDE model of the Encoder
    # model_decoder: Neural CDE model of the Decoder
    # validation_output: Tensor of output/treatment success, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # validation_toxic: Tensor of the side effects, size: number of patients x timepoints (in hours) x number of side effects
    # validation_treatments: Tensor of the treatments, size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables, size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    # static: Tensor of static variables: number of patients x number of variables, default: None
    
    # index: Index of variable
    # offset: Corresponds to start timepoint for computation of prediction measures
    # max_horizon: Corresponds to end timepoint for computation of prediction measures
    # loss: Indicating the used loss
    # rectilinear_index: Time index of covariables tensor, default: None
    # step: Corresponds to number of timestamps aggregated for computing the measures, default: None
    # unscaled: Inficator, whether variables should be normalized or not, default False
    # variables_std: list of the standard deviations of the variables, default None
    # variables_mean: list of the means of the variables, default None
    # variables: list of all variables
    
    # title: Title of the plot, default: str() empty string
    # save_link: Link to save computed figures, default: None
    # save_map: Link to save computed prediction measures, default: None
    # load_map: Link to load computed map, default: None
    # vmin: Minimum value of heatmap, default: None (using minimum observed datapoint)
    # vmax: Maximum value of heatmap, default: None (using maximum observed datapoint)
    # colorbar: Indicator, whether to plot colorbar or not default: True
    # diagmultistep: Indicator, whether 1-step diagonal is plotted or not default: False
    
    # dec_expand Indicating whether initialization of decoder is expanded (at least with static data) or not default: False
    # sofa_expand: Indicating whether initialization of decoder is expanded by last measured sofa_score default: False
    # med_dec: Indicating, whether Treatments are used as control or not default: False
    # med_dec_start: Indicating, whether initialization of decoder is expanded by the last measured treatment default: True
    
    
    # Loading data
    if load_map is not None:
        with open(load_map, 'rb') as handle:
            data2 = pickle.load(handle)
    
    # Computing prediction measured
    else:
        heat_data = np.zeros(shape=[max_horizon,max_horizon])
        heat_data[:] = np.nan
        
        if step is None or diagmultistep:
            mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = prediction_measures(model, data_map, validation_output, validation_toxic, validation_treatments, covariables.clone(), active_entries, static, unscaled=unscaled, rectilinear_index=rectilinear_index,variables_std=variables_std,variables_mean=variables_mean,variables=variables)
            
            if loss=='rmse':
                np.fill_diagonal(heat_data,rmse[offset:offset+max_horizon,index].cpu().detach().numpy())
            elif loss=='mse':
                np.fill_diagonal(heat_data,mse[offset:offset+max_horizon,index].cpu().detach().numpy())
            elif loss=='mae':
                np.fill_diagonal(heat_data,mae[offset:offset+max_horizon,index].cpu().detach().numpy())
            elif loss=='wape':
                np.fill_diagonal(heat_data,wape[offset:offset+max_horizon,index].cpu().detach().numpy())
            elif loss=='nrmse':
                np.fill_diagonal(heat_data,nrmse_sd[offset:offset+max_horizon,index].cpu().detach().numpy())
        
        for i in range(max_horizon-1):
            mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = prediction_measures_decoder(model,model_decoder, data_map,offset=offset,max_horizon=max_horizon,unscaled=unscaled, validation_output=validation_output, validation_toxic=validation_toxic, validation_treatments=validation_treatments, covariables=covariables.clone(), time_covariates=time_covariates, active_entries=active_entries, static=static, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=dec_expand, med_dec=med_dec, med_dec_start=med_dec_start)
            
            print(i)
            
            if loss=='rmse':
                heat_data[i,i+1:]=rmse[:-1,index].cpu().detach().numpy()
            elif loss=='mse':
                heat_data[i,i+1:]=mse[:-1,index].cpu().detach().numpy()
            elif loss=='mae':
                heat_data[i,i+1:]=mae[:-1,index].cpu().detach().numpy()
            elif loss=='wape':
                heat_data[i,i+1:]=wape[:-1,index].cpu().detach().numpy()
            elif loss=='nrmse':
                heat_data[i,i+1:]=nrmse_sd[:-1,index].cpu().detach().numpy()
            offset = offset+1
            max_horizon = max_horizon - 1
        
        data = np.ma.masked_invalid(heat_data)
        
        #ffill data with nans?!
        data2=ffill(data)
    
    if not invert:
        data2 = np.transpose(data2)
    
    # Creating plots
    if colorbar:
        fig, ax = plt.subplots(figsize=(5,4),dpi=400)
    else:
        fig, ax = plt.subplots(figsize=(4,4),dpi=400)
    
    if vmin is None:
        heatmap = ax.pcolor(data2, cmap=plt.cm.seismic, 
                            vmin=0, vmax=np.nanmax(data))
    else:
        heatmap = ax.pcolor(data2, cmap=plt.cm.seismic, 
                            vmin=vmin, vmax=vmax)
        
    # https://stackoverflow.com/a/16125413/190597 (Joe Kington)
    if colorbar:
        fig.colorbar(heatmap)
    
    # Setting some plotting options
    
    ax.set_xticks(np.arange(data2.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data2.shape[0])+0.5, minor=False)
    
    if max_horizon>72:
        a=np.arange(0,data2.shape[0],10)
        b=range(0,data2.shape[1],10)
    else:
        a=np.arange(0,data2.shape[0]+1,5)
        b=range(0,data2.shape[1]+1,5)
        
    ax.xaxis.set(ticks=a, ticklabels=a)

    ax.yaxis.set(ticks=b,ticklabels=b)
    
    if invert:
        ax.set_xlabel('Observed until treatment cycle x in months',fontsize= 13)
        ax.set_ylabel('Forecast horizon in treatment cycles',fontsize= 13)
    
    else:
        ax.set_ylabel('Observed until treatment cycle y in months',fontsize= 13)
        ax.set_xlabel('Forecast horizon in treatment cycles',fontsize= 13)
    
    
    if loss=='rmse':
        ax.set_title(title+'Root Mean Square Error')
    elif loss=='mse':
        ax.set_title(title+'Mean Square Error',fontsize= 13)
    elif loss=='mae':
        ax.set_title(title+'Mean Absolute Error')
    elif loss=='wape':
        ax.set_title(title+'WAPE')
    elif loss=='nrmse':
        ax.set_title(title+'NRMSE')
    
    if save_link is not None:
        plt.savefig(save_link)
    if save_map is not None:
        with open(save_map, 'wb') as handle:
            pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out