import torch.nn as nn
import torch

ms_error = nn.MSELoss()
ma_error = nn.L1Loss()

def disc_loss(pred , real = True):
    if real:
        target_tensor = torch.tensor(1.).to(pred.device)
    else:
        target_tensor = torch.tensor(0.).to(pred.device)
    
    target_tensor = target_tensor.expand_as(pred)

    return ms_error(pred,target_tensor)


def hidden_loss(hr,hf):
    l = len(hr)
    loss = 0
    for i in range(l):
        loss = loss + ma_error(hr[i],hf[i])
        
    return loss/l


def spec_loss(y,ypred,im_size = (256,512),ex = 1e-7):
    
    s = im_size[0]*im_size[1]
    
    lmag = torch.norm(torch.absolute(y) - torch.absolute(ypred), p=1)/s
    lsc = torch.norm(ypred-y, p = "fro")/(torch.norm(y, p = "fro")+ex)
    
    return lsc + lmag


def get_target_tensor(target, target_is_real):

    if target_is_real:
        target_tensor = torch.tensor(1.).to(target.device)
    else:
        target_tensor = torch.tensor(0.).to(target.device)
    return target_tensor.expand_as(target)

discriminator_loss = torch.nn.BCEWithLogitsLoss()
def n_layer_loss(target,state):
    loss = discriminator_loss(target,get_target_tensor(target,True))
    return loss


m = nn.LogSoftmax(dim=1)
def lm_loss(label,pred):
    pred = pred[:,:-1,:]
    pred = pred.reshape((pred.shape[0]*pred.shape[1],-1))

    label = label[:,1:]
    label = label.reshape(-1)
    
    loss = -1*m(pred)
    loss = torch.gather(loss, 1, torch.unsqueeze(label, 1))
    loss = loss
    return torch.sum(loss)/(pred.size()[0])

