import torch
from torch import nn 
import torchvision.models as models
import numpy as np
from copy import deepcopy

# only kaiming uniform or kaiming normal 
def initialize(model, initname ):
  didenter = False
  model.initname = initname
  for m in model.modules():     
    # weights    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if 'kaiming_normal' in initname:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')                
        didenter = True
      elif 'kaiming_uniform' in initname:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        didenter = True
      
      # elif 'xavier_normal' in initname:
      #   nn.init.xavier_normal_(m.weight)
      #   didenter = True
      # elif 'xavier_uniform' in initname:
      #   nn.init.xavier_uniform_(m.weight)
      #   didenter = True      

    # bias of linear layers
    if isinstance(m, nn.Linear) and (m.bias is not None):
      nn.init.constant_(m.bias, 0)

  if not didenter:
    raise Exception("Sorry, initname not in the list")

# negative network
def negative_model(model):
  net = deepcopy(model)
  state_dict = net.state_dict()
  for name, param in state_dict.items():
      # Transform the parameter as required.
      transformed_param = torch.mul(param, -1)
      # transformed_param = param * -1

      # Update the parameter.
      state_dict[name].copy_(transformed_param)
      
      net.initname = 'negative_' + model.initname
  # model.train()
  return net

def distorted_model(model):
  net = deepcopy(model)
  state_dict = net.state_dict() 

  for name, parm in state_dict.items():
    #distort the parameter
    transformed_param = parm + 1e-6

    #update the parameter
    state_dict[name].copy_(transformed_param)

    net.initname = 'distorted_' + model.initname 
  
  return net
