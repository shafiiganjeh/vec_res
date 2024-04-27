import torch
import torch.nn as nn
import torch.nn.functional as F
 
def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    w = l2norm(t)
    s_ = torch.matmul(w, w.T)
    K = s_.size()[0]
    l = torch.norm(s_ - torch.eye(K).to(s_.device))/(K**2)
    return l

class VectorQuantizer(nn.Module):

    def __init__(self,  num_embeddings = 8192, embedding_dim = 256, beta=0.25,train = True,l = 10,em_steps = 10):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.train = train
        self.l = l

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.embedding_dim)
        
        self.size = None
        self.x_perm = None
        self.em_steps = em_steps
        
        self.ind_select_vec = None
        self.ind_select_keep= None
            
        if train:
            self.em = 0
            self.dist = None
            self.worst = None
            self.ema = None
        
    def forward(self, x):

        self.size = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        self.x_perm = x.shape
        
        x_flattened = x.view(-1, self.embedding_dim)

        min_d = self.code_indices(x_flattened)
        one_hot = F.one_hot(min_d,self.num_embeddings).to(self.embedding.weight)
        
        x_q = torch.matmul(one_hot,self.embedding.weight)
        comp = x_q
        x_q = x_q.view(self.x_perm)
        
        if self.train:
            
            if self.worst is None:
                self.ema = torch.zeros(self.num_embeddings).to(x.device)
                self.worst = x_flattened.detach()
                self.dist = torch.mean((comp-x_flattened)**2,1).detach()
            else:
                min_w = self.code_indices(self.worst)
                one_hot_w = F.one_hot(min_w,self.num_embeddings).to(self.embedding.weight)
                comp_w = torch.matmul(one_hot_w,self.embedding.weight)

                self.dist = torch.mean((comp_w-self.worst)**2,1)
                dist = torch.mean((comp-x_flattened)**2,1).detach()
                rep = (self.dist - dist) < 0
                
                self.dist = rep*dist + (~rep)*self.dist
                self.worst = torch.einsum('ab,a->ab', x_flattened.detach(), rep) + torch.einsum('ab,a->ab', self.worst, ~rep)
            
            self.hit(torch.clip(torch.sum(one_hot, 0),0,1))
            
            
            loss = torch.mean((x_q.detach()-x)**2) + self.beta * \
                torch.mean((x_q - x.detach()) ** 2)
            loss = loss + self.l*orthogonal_loss_fn(self.embedding.weight.data)
            
        else:
            loss = None
        
        x_q = x + torch.sub(x_q, x).detach()        
        x_q = x_q.permute(0, 3, 1, 2).contiguous()    
        
        return x_q,loss,min_d.view(self.size[0],-1)

    def code_indices(self, flattened_inputs):

        similarity = torch.matmul(flattened_inputs, self.embedding.weight.t())
        
        d = (
                        torch.sum(flattened_inputs ** 2, dim=1, keepdim=True)
                        + torch.sum(self.embedding.weight**2, dim=1)
                        - 2 * similarity)

        min_d = torch.argmin(d, dim=1)
        
        return min_d
    
    def get_vec(self, tok):
        one_hot = F.one_hot(tok,self.num_embeddings).to(self.embedding.weight)
        x_q = torch.matmul(one_hot,self.embedding.weight).view(self.x_perm)
        x_q = x_q.permute(0, 3, 1, 2).contiguous()
        return x_q
    
    def hit(self, step):
        self.ema = torch.clip(step + self.ema,0,1)
        
        if self.em == self.em_steps:

            samples = torch.sum((self.ema < 1))
            self.ind_select_vec = torch.randperm(self.worst.size()[0])[:torch.clip(samples,0,self.worst.size()[0])]
            self.ind_select_keep = torch.randperm(samples)[torch.numel(self.ind_select_vec):]
            
            keep = self.embedding.weight.data[self.ema < 1,:][self.ind_select_keep,:]
            
            self.embedding.weight.data = torch.concat((keep,self.embedding.weight.data[self.ema > 0,:],self.worst[self.ind_select_vec,:]))
            self.em = 0
            self.ema = torch.zeros(self.num_embeddings).to(self.ema.device)
            
        self.em = self.em + 1
        return 
    

