import sys
import torch
import torch.nn as nn
import copy
sys.path.append("..")
from layers import transformer as l

#standard transformer with cross attention

class transformer(nn.Module):
    def __init__(self,n_embd = 768,n_head = None,train = True,n_ctx = None,code_block=None,n_layer = None,mask = False,cross = False,n_spe = 0):

            r"""
            Args:
                n_embd (int): embedding dimension.
                n_head (int): attention heads.
                train (bool): training mode.
                n_layer (int): number of transformer layers.
                n_ctx (int): sequence length.
                cross (bool): use cross attention.
                code_block (int): vocab. size.
                mask (bool): use unidirectional mask.
                n_spe (int): number of special tokens for finetuning.
            """

        super(transformer, self).__init__()
        
        self.train = train
        self.n_vocab = code_block + n_spe
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_spe = n_spe
        
        self.drop = nn.Dropout(p=0.1)
        
        wpe = torch.empty(n_ctx, self.n_embd)
        nn.init.normal_(wpe, std=0.01)
        self.wpe = torch.nn.parameter.Parameter(wpe)
        
        wte = torch.empty(code_block, self.n_embd)
        nn.init.normal_(wte, std=0.02)
        self.wte = torch.nn.parameter.Parameter(wte)
        
        if n_spe > 0:
            wte_sp = torch.empty(n_spe, self.n_embd)
            nn.init.normal_(wte_sp, std=0.02)
            self.wte_sp = torch.nn.parameter.Parameter(wte_sp).to(self.wte.device)
        else:
            wte_sp = torch.empty((0,self.n_embd))
            self.wte_sp = torch.nn.parameter.Parameter(wte_sp).to(self.wte.device)
        
        block_ = l.block(n_embd = self.n_embd,n_head = n_head,n_ctx = n_ctx,mask = mask,cross = cross,train = self.train)
        self.h = nn.ModuleList([copy.deepcopy(block_) for i in range(n_layer)])
        
        self.ln_f = l.norm(self.n_embd)
        
    def forward(self, x,y = torch.empty((0,0)) ,past = torch.empty((0,0))):
        results = {}
        batch, sequence = (x.size()[0],x.size()[1])
        
        past_length = past.size()[-2]
        wte = torch.cat((self.wte,self.wte_sp),dim = 0)

        
        if self.train:
            wte = self.drop(wte)
            wpe = self.drop(self.wpe)
        else:
            wpe = self.wpe
        

        h = wte.index_select(0, torch.flatten(x)) + wpe.index_select(0, (torch.arange(sequence,device= wpe.device) + past_length).repeat(batch))

        h = torch.reshape(h,(batch,sequence,-1)).contiguous()
        
        presents = []
        pasts = torch.unbind(past, dim = 1) if past.size()[-2] != 0 else [torch.empty((0,0))] * self.n_layer
        assert len(pasts) == self.n_layer
        
        for i, block in enumerate(self.h):
            h, present = block(x = h,y = y,past = pasts[i])
            presents.append(present)
            
        results['present'] = torch.stack(presents, dim=1)
        h = self.ln_f(h)
        
        h_flat = torch.reshape(h, (batch*sequence, self.n_embd))
        logits = torch.matmul(h_flat, torch.transpose(wte, 0, 1))
        logits = torch.reshape(logits, (batch, sequence, self.n_vocab))
        results['logits'] = logits
        return results
