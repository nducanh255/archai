# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import types 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from archai.nlp.nvidia_transformer_xl.nvidia_utils.log_uniform_sampler import LogUniformSampler
from archai.nlp.nvidia_transformer_xl.nvidia_utils.log_uniform_sampler import sample_logits
from archai.nlp.nvidia_transformer_xl.nvidia_utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.nlp.nvidia_transformer_xl import data_utils
from archai.common import utils, common


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)

torch_triu = torch.triu
def triu_onnx(x, diagonal=0):
    assert len(x.shape) == 2
    arange = torch.arange(x.size(0), device = x.device)
    arange2 = torch.arange(x.size(1), device = x.device)
    mask = arange.unsqueeze(-1).expand(-1, x.size(1)) <= (arange2 - diagonal)
    return x.masked_fill(mask==0, 0)
torch.triu = triu_onnx


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [bsz x n_head x qlen x klen]
        attn_score = torch.einsum('ibnd,jbnd->bnij', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # [bsz x n_head x qlen x klen] * [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('bnij,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += 'n_head={n_head},'
        return s.format(**self.__dict__)


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
    
    def extra_repr(self):
        s = super().extra_repr()
        s += 'n_head={n_head},'
        return s.format(**self.__dict__)

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask.bool()
        else:
            return mask.flip(0).bool()

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

        self.flops = 0

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        self.flops = 0
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        self.flops += torch.prod(torch.tensor(w_heads.size())) * w.size(-1)
        self.flops += torch.prod(torch.tensor(r_head_k.size())) * r.size(-1)
        
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', (rw_head_q, w_head_k))    # bsz x n_head x qlen x klen
        self.flops += torch.prod(torch.tensor(AC.size())) * w_head_k.size(-1)

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', (rr_head_q, r_head_k))     # bsz x n_head x qlen x klen
        self.flops += torch.prod(torch.tensor(BD.size())) * r_head_k.size(-1)
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', (attn_prob, w_head_v))
        self.flops += torch.prod(torch.tensor(attn_vec.size())) * attn_prob.size(-1)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        self.flops += torch.prod(torch.tensor(attn_out.size())) * attn_vec.size(-1)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        r_bias = r_bias.t()

        # compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                        # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->bnij', (rw_head_q, w_head_k))  # bsz x n_head x qlen x klen
        B_ = torch.einsum('ibnd,jnd->bnij', (w_head_q, r_emb))       # bsz x n_head x qlen x klen
        D_ = r_bias[None, :, None, :]                                # 1   x n_head x    1 x klen
        BD = self._rel_shift(B_ + D_)

        # [bsz x qlen x klen x n_head]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
                                                  dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout,
                                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList() #nn.ModuleList()

        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=(sample_softmax > 0)))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed).zero_()))
                # self.emb_projs.append(nn.Linear(d_embed, d_proj))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i).zero_()))
                # self.emb_projs.append(nn.Linear(d_emb_i, d_proj))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
                # embed = self.emb_projs[0](embed)
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i]).to(emb_flat.dtype)
                # emb_i = self.emb_projs[i](emb_i).to(emb_flat.dtype)

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        # the default attention
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # learnable embeddings
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # absolute embeddings
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            if tie_weight:
                emb_layers = [i.weight for i in self.word_emb.emb_layers]
            else:
                emb_layers = None

            emb_projs = self.word_emb.emb_projs
            # emb_projs = nn.ParameterList([i.weight for i in self.word_emb.emb_projs])

            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val,
                                                    tie_projs=tie_projs,
                                                    out_projs=emb_projs,
                                                    out_layers_weights=emb_layers)


        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        # default attention
        if self.attn_type == 0:
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
        # learnable
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head).zero_())
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head).zero_())
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head).zero_())
        # absolute standard
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        # absolute deeper SA
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.d_model).zero_())

    def reset_length(self, tgt_len, ext_len, mem_len):
        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems = torch.empty(self.n_layer, 0, dtype=param.dtype,
                               device=param.device)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            stacked = torch.stack(hids)
            if (
                self.mem_len == self.tgt_len
                and self.ext_len == 0
                and stacked.size(1) == self.mem_len
            ):
                new_mems = stacked.detach()
            else:
                end_idx = mlen + max(0, qlen - self.ext_len)
                beg_idx = max(0, end_idx - self.mem_len)
                if mems.numel():
                    cat = torch.cat([mems, stacked], dim=1)
                else:
                    cat = stacked
                new_mems = cat[:, beg_idx:end_idx].detach()

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len - 1
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).bool()
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

        hids = []
        # default
        if self.attn_type == 0:
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                                 self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
        # learnable
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        # absolute
        elif self.attn_type == 2:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    def forward(self, data, target, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if mems is None:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                  pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)
        
        return (loss, new_mems)


class MemTransformerLM_flex(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        super(MemTransformerLM_flex, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_heads = n_head
        self.d_heads = [d_model//n_head for n_head in n_head] #d_heads
        assert (np.multiply(self.d_heads, self.n_heads)==[self.d_model]*len(self.n_heads)).all(), "d_model must be divisible by sampled num_heads"

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        # the default attention
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(n_head[i], d_model, self.d_heads[i], d_inner[i], dropout, tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                                    dropatt=dropatt, pre_lnorm=pre_lnorm))
        # learnable embeddings
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(n_head[i], d_model, self.d_heads[i], d_inner[i], dropout, tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                            dropatt=dropatt, pre_lnorm=pre_lnorm))
        # absolute embeddings
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(DecoderLayer(n_head[i], d_model, self.d_heads[i], d_inner[i], dropout, dropatt=dropatt, pre_lnorm=pre_lnorm))

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            if tie_weight:
                emb_layers = [i.weight for i in self.word_emb.emb_layers]
            else:
                emb_layers = None

            emb_projs = self.word_emb.emb_projs
            # emb_projs = nn.ParameterList([i.weight for i in self.word_emb.emb_projs])

            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val,
                                                    tie_projs=tie_projs,
                                                    out_projs=emb_projs,
                                                    out_layers_weights=emb_layers)


        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        # default attention
        if self.attn_type == 0:
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = []
            self.r_r_bias = []
            for i, layer in enumerate(self.layers):
                self.r_w_bias.append(nn.Parameter(torch.Tensor(self.n_heads[i], self.d_heads[i]).zero_()))
                self.r_r_bias.append(nn.Parameter(torch.Tensor(self.n_heads[i], self.d_heads[i]).zero_()))
        # learnable
        elif self.attn_type == 1:
            self.r_emb = []
            self.r_w_bias = []
            self.r_bias = []
            for i, layer in enumerate(self.layers):
                self.r_emb.append(nn.Parameter(torch.Tensor(self.max_klen, self.n_heads[i], self.d_heads[i]).zero_()))
                self.r_w_bias.append(nn.Parameter(torch.Tensor(self.n_heads[i], self.d_heads[i]).zero_()))
                self.r_bias.append(nn.Parameter(torch.Tensor(self.max_klen, self.n_heads[i]).zero_()))
        # absolute standard
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        # absolute deeper SA
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.d_model).zero_())

    def reset_length(self, tgt_len, ext_len, mem_len):
        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems = torch.empty(self.n_layer, 0, dtype=param.dtype,
                               device=param.device)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            stacked = torch.stack(hids)
            if (
                self.mem_len == self.tgt_len
                and self.ext_len == 0
                and stacked.size(1) == self.mem_len
            ):
                new_mems = stacked.detach()
            else:
                end_idx = mlen + max(0, qlen - self.ext_len)
                beg_idx = max(0, end_idx - self.mem_len)
                if mems.numel():
                    cat = torch.cat([mems, stacked], dim=1)
                else:
                    cat = stacked
                new_mems = cat[:, beg_idx:end_idx].detach()

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len - 1
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len)).bool()
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

        hids = []
        # default
        if self.attn_type == 0:
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias[i],
                                 self.r_r_bias[i], dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
        # learnable
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        # absolute
        elif self.attn_type == 2:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    def forward(self, data, target, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if mems is None:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                  pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)
        
        return (loss, new_mems)


def forward_predict_memtransformer(self, data):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
    tgt_len = data.size(0)
    hidden, _ = self._forward(data, mems=None)

    pred_hid = hidden[-tgt_len:]
    out = self.crit(pred_hid.view(-1, pred_hid.size(-1)))
    out = out.view(tgt_len, -1)
    
    return out

# provides a full log probability over the entire vocab
def predict(self, hidden):
    '''
        hidden :: [len*bsz x d_proj]
    '''
    self.flops = 0

    if self.n_clusters == 0:
        logit = self._compute_logit(hidden, self.out_layers_weights[0], self.out_layers_biases[0], self.get_out_proj(0))
        output = torch.argmax(logit, dim=1)
    else:
        # construct weights and biases
        weights, biases, projs = [], [], []
        for i in range(len(self.cutoffs)):
            if self.div_val == 1:
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                weight_i = self.out_layers_weights[0][l_idx:r_idx]
                bias_i = self.out_layers_biases[0][l_idx:r_idx]
            else:
                weight_i = self.out_layers_weights[i]
                bias_i = self.out_layers_biases[i]
            projs.append(self.get_out_proj(i))

            if i == 0:
                weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

            weights.append(weight_i)
            biases.append(bias_i)

        head_weight, head_bias, head_proj = weights[0], biases[0], projs[0]
        head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
        output = torch.argmax(head_logit, dim=1)
        not_in_shortlist = (output >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any()) 
        
        # log_prob = self._get_full_log_prob(hidden, head_logit, weights[1:], biases[1:], projs[1:])
        
        if all_in_shortlist:
            # pass
            return output
        elif not_in_shortlist.all():
            print('option 2')
            log_prob = self._get_full_log_prob(hidden, head_logit, weights[1:], biases[1:], projs[1:])
            return torch.argmax(log_prob, dim=1)   
        else:
            print('option 3')
            log_prob = self._get_full_log_prob(hidden[not_in_shortlist], head_logit[not_in_shortlist], weights[not_in_shortlist], biases[not_in_shortlist], projs[not_in_shortlist])
            output[not_in_shortlist] = torch.argmax(log_prob[not_in_shortlist], dim=1)
            return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=6, help='')
    parser.add_argument('--n_token', type=int, default=267735, help='')
    parser.add_argument('--n_head', type=lambda s: [int(item) for item in s.split(',')], default=[4], help='')
    parser.add_argument('--d_head', type=lambda s: [int(item) for item in s.split(',')], default=[32], help='')
    parser.add_argument('--d_model', type=int, default=128, help='')
    parser.add_argument('--d_embed', type=int, default=256, help='')
    parser.add_argument('--d_inner', type=lambda s: [int(item) for item in s.split(',')], default=[1204], help='')
    parser.add_argument('--div_val', type=int, default=1, help='') # Dividend value for adaptive input and softmax
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')
    parser.add_argument('--setup_only', action='store_true', help='')

    args = parser.parse_args()

    tgt_len, mem_len, ext_len = 192, 0, 0
    cutoffs = [19997, 39997, 199997] #[args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    if not args.setup_only:
        if len(args.n_head)==1:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head[0],
                                        args.d_model, args.d_head[0], args.d_inner[0],
                                        args.dropout, dropatt=args.dropout,
                                        tie_weight=True, d_embed=args.d_embed,
                                        div_val=args.div_val, tie_projs=tie_projs,
                                        pre_lnorm=True, tgt_len=tgt_len,
                                        ext_len=ext_len, mem_len=mem_len,
                                        cutoffs=cutoffs, attn_type=0,
                                        dtype=None)
        else:
            model = MemTransformerLM_flex(args.n_token, args.n_layer, args.n_head,
                                        args.d_model, args.d_head, args.d_inner,
                                        args.dropout, dropatt=args.dropout,
                                        tie_weight=True, d_embed=args.d_embed,
                                        div_val=args.div_val, tie_projs=tie_projs,
                                        pre_lnorm=True, tgt_len=tgt_len,
                                        ext_len=ext_len, mem_len=mem_len,
                                        cutoffs=cutoffs, attn_type=0,
                                        dtype=None)
        model.forward = types.MethodType(forward_predict_memtransformer, model)
        model.crit.forward = types.MethodType(predict, model.crit)
        # print(model)

        device = torch.device("cuda" if args.cuda else "cpu")
        model = model.to(device)
        model.eval()

        print('# total params', sum(p.numel() for p in model.parameters()))
        print('# embd params', sum(p.numel() for p in model.word_emb.parameters()))
        print('# layer params', sum(p.numel() for p in model.layers[0].parameters()))

        # sample run
        B = 1 # batch size
        data_len = tgt_len
        data = torch.LongTensor(data_len*B).random_(0, args.n_token).unsqueeze(-1).to(device)
        # for _ in range(1000):
        output = model(data)
        # torch.onnx.export(model, inp, os.path.join('onnx_models', 'memformer.onnx'), opset_version=13)
        print('done')

        # path_to_data = common.default_dataroot()
        # path_to_data = utils.full_path(os.path.join(path_to_data,'textpred', exp_utils.dataset_dir_name('wt103')))
        # corpus = get_lm_corpus(path_to_data, 'wt103', 'word', max_size=None)
        # diter = corpus.get_iterator('train', 256, 192, device='cpu', ext_len=0)
        # for idx, (inp, tgt, seqlen, _) in enumerate(diter):
        #     output = model(inp)
        #     torch.onnx.export(model, inp, os.path.join('onnx_models', 'memformer.onnx'), opset_version=13)
        #     break
