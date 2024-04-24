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

    def __init__(self,  num_embeddings, embedding_dim, beta=0.25,train = True,l = 10):
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

    def forward(self, x):
        self.size = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        self.x_perm = x.shape
        
        x_flattened = x.view(-1, self.embedding_dim)

        min_d = self.code_indices(x_flattened)
        one_hot = F.one_hot(min_d,self.num_embeddings).to(self.embedding.weight)
        
        x_q = torch.matmul(one_hot,self.embedding.weight).view(self.x_perm)
        
        if self.train:
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
    
