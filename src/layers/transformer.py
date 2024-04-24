import torch
import torch.nn as nn


class conv1d(nn.Module):
    def __init__(self, n_in, n_out,w_init_stdev=0.02):
        super(conv1d, self).__init__()
        self.n_out = n_out
        self.n_in = n_in
        
        w = torch.empty( 1,n_in, n_out)
        nn.init.normal_(w, std=w_init_stdev)
        
        self.w = torch.nn.parameter.Parameter(w)
        self.b = torch.nn.parameter.Parameter(torch.zeros(n_out))
        
    def forward(self, x):
        out_shape = x.size()[:-1] + (self.n_out,)
        w = torch.reshape(self.w, (-1, self.n_out))
        x = torch.reshape(x, (-1, self.n_in))
        c = torch.matmul(x, w) + self.b
        return torch.reshape(c,out_shape)


class attn(nn.Module):
    def __init__(self,  n_embd = None,n_head = None,n_ctx = None,cross = False,mask = True,train = True):
        super(attn, self).__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.mask = mask
        self.cross = cross
        self.train = train
        
        self.drop = nn.Dropout(p=0.1)
        
        if cross == True:
            self.c_attn_cross_kv = conv1d(n_embd, n_embd * 2)
            self.c_proj_cross = conv1d(n_embd, n_embd)
        else:
            self.c_attn = conv1d(n_embd, n_embd * 3)
            self.c_proj = conv1d(n_embd, n_embd)

        self.softmax = torch.nn.Softmax(dim = -1)
        
        self.register_buffer("mask_", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        
    def mask_attn_weights(self,w):
        nd, ns = w.size(-2), w.size(-1)
        mask = self.mask_[:, :, ns-nd:ns, :ns]
        w = w*mask + -1e9*(1-mask)
        return w

    def split_states(self,x, n = None):
        shape = x.size()
        return torch.reshape(x, (shape[0],shape[1],-1 ,shape[-1]//n))

    def split_heads(self,x):
        return torch.permute(self.split_states(x, torch.tensor(self.n_head)), (0, 2, 1, 3))

    def merge_states(self,x):
        shape = x.size()
        x = torch.permute(x, (0, 2, 1, 3))
        return torch.reshape(x, (shape[0],shape[2],-1))

    def multihead_attn(self,q, k, v):
        k = torch.transpose(k, 2, 3)
        w = torch.matmul(q,k)
        w = w * torch.rsqrt(torch.tensor(v.size()[-1]))
        if self.mask == True:
            w = self.mask_attn_weights(w)
        w = self.softmax(w)
        a = torch.matmul(w, v)
        return a

    def forward(self,x,y = torch.empty((0,0)) , past = torch.empty((0,0))):
        if self.cross:
            q = x
            c = self.c_attn_cross_kv(y)
            k, v = torch.split(c, self.n_embd, dim = 2)
            q = self.split_heads(q)
            k = self.split_heads(k)
            v = self.split_heads(v)
            
            a = self.multihead_attn(q, k, v)
            a = self.merge_states(a)
            a = self.c_proj_cross(a)
            if self.train:
                a = self.drop(a)
            present = past
        else:
            c = self.c_attn(x)
            q, k, v = torch.split(c, self.n_embd, dim = 2)
            q = self.split_heads(q)
            k = self.split_heads(k)
            v = self.split_heads(v)
            
            if past.numel() != 0:
                pk, pv = torch.unbind(past, dim=1)
                k = torch.cat((pk, k), -2)
                v = torch.cat((pv, v), -2)
                
            present = torch.stack((k,v), dim = 1)
                
            a = self.multihead_attn(q, k, v)
            a = self.merge_states(a)
            a = self.c_proj(a)
            if self.train:
                a = self.drop(a)
        return a,present
    

class mlp(nn.Module):
    def __init__(self, n_in, n_state,train = True):
        super(mlp, self).__init__()
        self.c_fc = conv1d(n_in,n_state)
        self.c_proj = conv1d(n_state,n_in)
        self.gelu = torch.nn.functional.gelu
        self.drop = nn.Dropout(p=0.1)
        self.train = train
        
    def forward(self, x):
        h = self.gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        if self.train:
            h2 = self.drop(h2)
        return h2


class norm(nn.Module):
    def __init__(self, n_state, epsilon=1e-12):
        super(norm, self).__init__()
        
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.epsilon = torch.tensor(epsilon)
        
    def forward(self, x):
        u = torch.mean(x, -1, keepdim=True)
        s = torch.square(x-u)
        s = torch.mean(s, -1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.epsilon)
        x = x * self.g + self.b
        return x


class block(nn.Module):
    def __init__(self,  n_embd = None,n_head = None,n_ctx = None,mask = True,cross = False,train = True):
        super(block, self).__init__()
        self.train = train 
        self.mlp = mlp(n_embd,4 * n_embd,train = self.train)
        self.attn = attn(n_embd = n_embd,n_head = n_head,n_ctx = n_ctx,cross = False,mask = mask,train = self.train)
        self.ln_1 = norm(n_embd)
        self.ln_2 = norm(n_embd)
        self.cross = cross 
        self.mask = mask
        
        if cross == True:
            self.attn_cross = attn(n_embd = n_embd,n_head = n_head,n_ctx = n_ctx,mask = False,cross = True)
            self.ln_cross = norm(n_embd)

    def forward(self, x,y =  torch.empty((0,0)),past = torch.empty((0,0))):
        if self.cross == True:
            a, present = self.attn(x = self.ln_1(x),past = past)
            x = x + a
            a, present_ = self.attn_cross(x = self.ln_cross(x),y = y)
            x = x + a
            m = self.mlp(self.ln_2(x))
            x = x + m
        else:
            a, present = self.attn(x = self.ln_1(x),past = past)
            x = x + a
            m = self.mlp(self.ln_2(x))
            x = x + m
        return x, present