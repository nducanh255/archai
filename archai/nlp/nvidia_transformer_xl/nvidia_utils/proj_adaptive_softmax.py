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

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionalParameterList(nn.ParameterList):
    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            if p is not None:
                size_str = 'x'.join(str(size) for size in p.size())
                device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
                parastr = 'Parameter containing: [{} of size {}{}]'.format(
                    torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 tie_projs=None, out_layers_weights=None, out_projs=None,
                 keep_order=False):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        #self.cutoffs = cutoffs + [n_token]
        self.cutoffs = cutoffs
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.tie_projs = tie_projs

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        if not out_layers_weights:
            self.out_layers_weights = nn.ParameterList()
        else:
            self.out_layers_weights = out_layers_weights

        self.out_layers_biases = nn.ParameterList()

        self.shared_out_projs = out_projs
        self.out_projs = OptionalParameterList()

        if div_val == 1:
            if d_proj != d_embed:
                for i in range(len(self.cutoffs)):
                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(
                            nn.Parameter(torch.zeros(d_proj, d_embed))
                        )
            else:
                # self.out_projs = [None] * len(self.cutoffs)
                self.out_projs.append(None)

            self.out_layers_biases.append(
                nn.Parameter(torch.zeros(n_token))
                )

            if not out_layers_weights:
                self.out_layers_weights.append(
                    nn.Parameter(torch.zeros(n_token, d_embed))
                    )
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                if tie_projs[i]:
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(
                        nn.Parameter(torch.zeros(d_proj, d_emb_i))
                    )

                self.out_layers_biases.append(
                    nn.Parameter(torch.zeros(r_idx - l_idx))
                    )
                if not out_layers_weights:
                    self.out_layers_weights.append(
                        nn.Parameter(torch.zeros(r_idx - l_idx, d_emb_i))
                        )

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            if bias is not None:
                logit = logit + bias
        return logit

    def get_out_proj(self, i):
        if self.tie_projs[i]:
            if len(self.shared_out_projs) == 0:
                return None
            elif len(self.shared_out_projs) == 1:
                return self.shared_out_projs[0]
            else:
                return self.shared_out_projs[i]
        else:
            return self.out_projs[i]

    def _forward_std(self, hidden, target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))
            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                if self.keep_order or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll

    def forward_kd(self, hidden, target, soft_target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
            soft_target :: (values, indices): ([len*bsz, k], [len*bsz, k])
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))
            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    inds_i = soft_target[1]
                    prob_kd = soft_target[0]
                    logporb_kd = head_logprob_i.gather(1, inds_i)
                    kl_div = prob_kd * (torch.log(prob_kd) - logporb_kd)
                    kd_loss = torch.sum(kl_div, dim=1)
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    tail_logit = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                    tail_logprob = F.log_softmax(tail_logit, dim=1)

                    tail_logprob_i = tail_logprob.index_select(0, indices_i)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                if self.keep_order or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll

    def forward(self, hidden, target, soft_target, keep_order=False):

        if soft_target:
            return self._forward_kd(hidden, target, soft_target, keep_order)
        else:
            nll = self._forward_std(hidden, target, keep_order)
            return nll, nll

    def _forward_kd(self, hidden, target, soft_target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
            soft_target :: (values, indices): ([len*bsz, k], [len*bsz, k])
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))
            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)


            n_sample_chunks = 5
            n_samples = hidden.size(0)
            chunk_sizes = [n_samples // n_sample_chunks] * n_sample_chunks
            chunk_sizes[-1] += n_samples % n_sample_chunks
            chunk_offset = 0

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            loss = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            temp = 1
            alpha = 0.5

            for chunk_n in range(n_sample_chunks):
                chunk_size = chunk_sizes[chunk_n]
                hidden_chunk = hidden[chunk_offset:chunk_offset+chunk_size]

                head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

                head_logit = self._compute_logit(hidden_chunk, head_weight, head_bias, head_proj)
                head_logprob = F.log_softmax(head_logit / temp, dim=1)
                # head_logprob = head_logit

                offset = head_logprob.size(1) - self.n_clusters
                logprob = torch.zeros((hidden_chunk.size(0), self.cutoffs[-1]), dtype=hidden.dtype, device=hidden.device)
                logprob[:, 0:offset] = head_logprob[:, 0:offset]

                cutoff_values = [0] + self.cutoffs
                for i in range(1, len(cutoff_values) - 1):

                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    tail_logit = self._compute_logit(hidden_chunk, weight_i, bias_i, proj_i)
                    tail_logprob = F.log_softmax(tail_logit / temp, dim=1)
                    # tail_logprob = tail_logit

                    # Is this right?
                    logprob_i = head_logprob[:, -i][:, None] + tail_logprob

                    logprob[:, offset:offset+logprob_i.size(1)] = logprob_i

                    offset += logprob_i.size(1)

                nll_chunk = -logprob.gather(1, target[chunk_offset:chunk_offset+chunk_size, None]).squeeze(1)
                # nll[chunk_offset:chunk_offset+chunk_size] = (1.0 - alpha) * nll_chunk
                nll[chunk_offset:chunk_offset+chunk_size] = nll_chunk


                if soft_target:
                    inds = soft_target[1][chunk_offset:chunk_offset+chunk_size]
                    prob_kd = soft_target[0][chunk_offset:chunk_offset+chunk_size]
                    logprob_kd = logprob.gather(1, inds)
                    # kl_div = prob_kd * (torch.log(prob_kd) - logprob_kd)
                    kl_div = prob_kd * logprob_kd
                    # kl_div = F.kl_div(logprob_kd, prob_kd, reduction='none')
                    kd_loss = -torch.sum(kl_div, dim=1)

                    loss[chunk_offset:chunk_offset+chunk_size] = alpha * kd_loss * (temp*temp) + (nll_chunk * (1 - alpha))
                else:
                    loss[chunk_offset:chunk_offset+chunk_size] = nll_chunk

                chunk_offset += chunk_size

        return (nll, loss)

    def proba(self, hidden):
        '''
            hidden :: [len*bsz x d_proj]
        '''

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))
            proba = F.softmax(logit, dim=-1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            results_val = []
            results_ind = []

            n_sample_chunks = 5
            n_samples = hidden.size(0)
            chunk_sizes = [n_samples // n_sample_chunks] * n_sample_chunks
            chunk_sizes[-1] += n_samples % n_sample_chunks
            chunk_offset = 0
            temp = 1

            for chunk_n in range(n_sample_chunks):
                chunk_size = chunk_sizes[chunk_n]
                hidden_chunk = hidden[chunk_offset:chunk_offset+chunk_size]

                head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

                head_logit = self._compute_logit(hidden_chunk, head_weight, head_bias, head_proj)
                head_prob = F.softmax(head_logit / temp, dim=1)

                offset = head_prob.size(1) - self.n_clusters
                proba = torch.zeros((hidden_chunk.size(0), self.cutoffs[-1]), dtype=hidden.dtype, device=hidden.device)
                proba[:, 0:offset] = head_prob[:, 0:offset]

                cutoff_values = [0] + self.cutoffs
                for i in range(1, len(cutoff_values) - 1):
                    l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    tail_logit_i = self._compute_logit(hidden_chunk, weight_i, bias_i, proj_i)
                    tail_prob_i = F.softmax(tail_logit_i / temp, dim=1)

                    #prob_i = (head_prob[:, -i] * tail_prob_i.T).T
                    # prob_i = torch.einsum('i,ij->ij', (head_prob[:, -i], tail_prob_i))
                    prob_i = head_prob[:, -i][:, None] *  tail_prob_i

                    proba[:, offset:offset+prob_i.size(1)] = prob_i

                    offset += prob_i.size(1)

                vals, indices = torch.topk(proba, k=30, dim=-1)
                results_val.append(vals)
                results_ind.append(indices)
                chunk_offset += chunk_size

        return (torch.vstack(results_val), torch.vstack(results_ind))
