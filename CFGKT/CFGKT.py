import torch
from torch import nn
from torch.nn.init import xavier_uniform_,kaiming_normal_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
import numpy as np
import seaborn

class CFGKT(nn.Module):
    def __init__(self, n_concepts, n_pid, d_model, n_blocks,
                 kq_same, dropout, model_type, memory_size,final_fc_dim,
                 n_heads, d_ff,time,interval, separate_qa=False,attn_=True):
        super().__init__()
        '''
        CFGKT consists of the state-enhanced embedding, the coarse-grained block, the fine-grained block, and 
        the state-fusion attention. 
        '''
        self.n_question = n_concepts
        self.dropout = dropout
        self.kq_same = kq_same  # Whether k and q come from the same learning sequence.
        self.n_pid = n_pid  # Represent the answers as two vectors or a embedding matrix.
        self.model_type = model_type
        self.separate_qa = separate_qa
        self.time = time
        self.interval = interval
        self.d_model = d_model
        self.attn_ = attn_

        # Whether to represent the discrepancy in exercises and answers.
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, self.d_model)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, self.d_model)

        self.q_embed = nn.Embedding(self.n_question + 1, self.d_model)
        self.qa_embed = nn.Embedding(2 * self.n_question + 2, self.d_model)
        self.interval_embed = nn.Embedding(self.interval + 2,self.d_model,padding_idx=self.interval + 1)
        self.time_embed = nn.Embedding(self.time + 2,self.d_model,padding_idx=self.time + 1)
        self.decoder_map = nn.Linear(self.d_model * 3,self.d_model)
        self.encoder_map = nn.Linear(self.d_model * 3,self.d_model)

        self.unit = exp_acq(dropout,self.d_model)
        self.rnn_attn_map = nn.Linear(self.d_model * 2, self.d_model)
        self.trans = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads,
                                  dropout=dropout, d_model=self.d_model, d_feature=self.d_model / n_heads,
                                  d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type)
        self.rnn_chose = nn.Linear(self.d_model, self.d_model)
        self.attn_chose = nn.Linear(self.d_model, self.d_model)

        # initializing a matrix to embed the underlying latent knowledge concepts.
        self.MS = nn.Parameter(torch.Tensor(memory_size, self.d_model))
        kaiming_normal_(self.MS)

        self.state_attn = state_fusion(self.d_model, n_heads, device)
        if self.attn_ == True:
            self.final_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.rnn_a = nn.Linear(self.d_model, self.d_model)
        self.attn_a = nn.Linear(self.d_model, self.d_model)
        self.cgm = ext_mem(self.d_model, memory_size)
        self.fgm = ext_mem(self.d_model, memory_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model + self.d_model, final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(final_fc_dim, 256),
            nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(256, 1))

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, qa_data, target, pid_data,dotime, lagtime):
        q_embed_data = self.q_embed(q_data)

        # Embed students' answers.
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = (qa_data-q_data) // self.n_question
            qa_embed_data = self.qa_embed(qa_data)+q_embed_data

        # Embed the discrepancy over exercises and answers via the Rasch model.
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)

        # Embed the time-related information and concatenate them with answers and exercises.
        in_dotime = self.time_embed(dotime)
        in_lagtime = self.interval_embed(lagtime)
        if self.separate_qa:
            q_embed_data = q_embed_data
        else:
            q_embed_data = self.encoder_map(torch.cat((q_embed_data, in_dotime, in_lagtime),dim=-1))
        qa_embed_data = self.decoder_map(torch.cat((qa_embed_data, in_dotime, in_lagtime),dim=-1))

        # This is the process of CGB for proficiency.
        trans_output = self.trans(q_embed_data, qa_embed_data)

        # This is the prosess of FGB for experience.
        qa_shift = self.qa_embed(torch.LongTensor([2 * self.n_question])).repeat(qa_embed_data.size(0),1,1)
        qa_shift = torch.cat((qa_shift, qa_embed_data[:, :-1, :]), dim=1)
        unit_out = self.unit(q_embed_data, qa_embed_data)

        # Write & Read the proficiency and experience with the external memories.
        trans_output = self.cgm(q_embed_data, trans_output, self.MS)
        unit_out = self.fgm(q_embed_data, unit_out, self.MS)

        # This is the state-fusion attention mechanism.
        mask = ut_mask(unit_out.size(1))
        trans_output = self.state_attn(unit_out, trans_output, qa_shift)
        if self.attn_ == True: 
            trans_output, _ = self.final_attn(unit_out, trans_output, qa_shift, attn_mask=mask)

        # Predict students' performance on current question via the mlp layer.
        concat_q = torch.cat([trans_output, q_embed_data], dim=-1)
        output = self.mlp(concat_q)
        x = torch.sigmoid(output)
        return x.squeeze(-1)

class state_fusion(nn.Module):
    def __init__(self, hid_dim, n_heads, device):
        super(state_fusion, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
        self.attn_distilling = distilling(hid_dim,device)
        self.device = device

    def _reset_parameters(self):
        xavier_uniform_(self.w_q.weight)
        xavier_uniform_(self.w_k.weight)
        xavier_uniform_(self.w_v.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            constant_(self.q_linear.bias, 0.)

    def forward(self, query, key, value):
        bsz = query.shape[0]
        Q = self.w_q(query).to(self.device)
        K = self.w_k(key).to(self.device)
        V = self.w_v(value).to(self.device)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        mask = np.triu(np.ones((Q.size(-2), Q.size(-2))), k=1).astype("bool")
        mask = torch.from_numpy(mask).to(self.device)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = attention.masked_fill(mask, -1e10)
        masked_scores = torch.softmax(attention, dim=-1)

        # Compare with "Base" first, then compute the dot product.
        distilling_output = self.attn_distilling(masked_scores,V)
        # output = distilling_output.reshape(bsz,distilling_output.size()[-2],-1)
        x = distilling_output.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class distilling(nn.Module):
    def __init__(self, d_model, device):
        super(distilling, self).__init__()
        self.d_model = d_model
        self.device = device
        # self.parameter = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.para_matrix = nn.Parameter(torch.randn(64, 4, 500, 500),requires_grad=True)

    def _reset_parameters(self):
        xavier_uniform_(self.w_q.weight)
        xavier_uniform_(self.w_k.weight)
        xavier_uniform_(self.w_v.weight)
        constant_(self.w_q.bias, 0.)
        constant_(self.w_k.bias, 0.)
        constant_(self.w_v.bias, 0.)

    def average_V(self, V):
        B, H, n_q, d_q = V.shape
        V_mask = torch.triu(torch.ones(n_q, n_q)).expand(B, H, n_q, n_q)
        V_sum = torch.matmul(V.permute(0,1,3,2), V_mask).permute(0,1,3,2)
        V_average = torch.div(V_sum, torch.arange(1, n_q + 1).unsqueeze(-1).expand(B, H, n_q, 1))
        return V_average

    def locate_V(self, scores, V):
        B, H, n_q, _ = scores.shape
        # print(self.para_matrix[0])
        # print(self.para_matrix[1])
        base_score = torch.tril(torch.div(torch.ones(n_q), torch.arange(1, n_q + 1))
                                .unsqueeze(-2)
                                .expand(B, H, n_q, n_q).permute(0,1,3,2))
        # print_pra = torch.sigmoid(self.parameter)
        # print(print_pra[0])
        minus_score = (scores - base_score)
        bigger_matrix = torch.tril(torch.where(minus_score >= 0, scores, 0))
        # smaller_matrix = torch.tril(torch.where(minus_score < 0, 0.0, 0))
        # V_average = self.average_V(V)
        attn = torch.matmul(bigger_matrix, V)
        return attn

    def forward(self, masked_scores, v):
        state_h = self.locate_V(masked_scores, v)
        return state_h


def ut_mask(seq_len):
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)

class exp_acq(nn.Module):
    def __init__(self, dropout, d_model):
        super(exp_acq, self).__init__()
        '''
        This experience-acquisition unit is designed to assess the temporary problem-solving experience.
        '''
        self.d_model = d_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.exp = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.d_model)), requires_grad=True)  # experience
        self.linear_gate = nn.Linear(2 * self.d_model, self.d_model)    # gate_t
        self.linear_k = nn.Linear(2 * self.d_model, self.d_model)   # k_qt
        self.linear_re = nn.Linear(2 * self.d_model, self.d_model)  # reset gate

    def forward(self, q_embed_data, qa_embed_data):
        B,L,d = qa_embed_data.shape
        padding = torch.zeros(B, 1, d, device=self.device)
        a_emb = torch.cat((padding, qa_embed_data[:, :-1, :]), 1)
        qa = a_emb.split(1, dim=1)
        q_embed_data = torch.cat((padding, q_embed_data[:, :-1, :]), 1)
        q = q_embed_data.split(1, dim=1)
        exp = self.exp.repeat(q_embed_data.size(0), 1).cuda()
        h = list()
        h.append(exp.unsqueeze(1))
        seqlen = 99
        for i in range(1, seqlen+1):
            a_1 = torch.squeeze(qa[i], 1)
            an_input = torch.squeeze(q[i], 1)
            x = torch.cat((a_1, an_input),dim=-1)
            gate = self.sigmoid(self.linear_gate(x))
            k_q = self.tanh(self.linear_k(x))
            gain = gate * k_q
            ins = torch.cat((a_1, an_input), -1)
            reset = self.sigmoid(self.linear_re(ins))
            exp = reset * exp + (1 - reset) * gain
            h_i = torch.unsqueeze(exp, dim=1)
            h.append(h_i)
        temporary_ps = torch.cat(h, axis=1)
        return temporary_ps

class ext_mem(nn.Module):
    def __init__(self, d_model, m_size):
        super(ext_mem, self).__init__()
        '''
        This is the external memory for storing students' experience and proficiency.
        '''
        self.d_model = d_model
        self.m_size = m_size
        self.dy_mem = nn.Parameter(torch.Tensor(self.m_size, self.d_model))    # Initialize a dynamic memory matrix.
        kaiming_normal_(self.dy_mem)
        self.e_gate = nn.Linear(self.d_model, self.d_model)    # erase gate
        self.a_gate = nn.Linear(self.d_model, self.d_model)    # add gate

    def forward(self, q, qa, MS):
        # MS is the static matrix, storing the underlying latent knowledge concepts.
        batch_size = q.shape[0]
        mem_slot = self.dy_mem.unsqueeze(0).repeat(batch_size, 1, 1)
        mem = [mem_slot]
        w = torch.softmax(torch.matmul(q, MS.T), dim=-1)
        e = torch.sigmoid(self.e_gate(qa))
        a = torch.tanh(self.a_gate(qa))
        # Split first, splice second.
        for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
            mem_slot = mem_slot * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
            mem.append(mem_slot)
        final_mem = torch.stack(mem, dim=1)
        return (w.unsqueeze(-1) * final_mem[:, :-1]).sum(-2)

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super(Architecture, self).__init__()
        '''
        This is a transformer layer with n_blocks encoders and decoders.
        '''
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'CFGKT'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)    # Decoders
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks * 2)    # Encoders and the cross-attention
            ])

    def forward(self, q_embed_data, qa_embed_data):
        y = qa_embed_data
        x = q_embed_data
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first: 
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else: 
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        '''
        The encoders or decoders is composed of the multi-head self-attention, the Add & Norm, and the FFN.
        '''
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0)
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, gammas)

        # Concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)

def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # Scores does not participate in backpropagation.
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()
        return torch.matmul(scores_, v)

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]

class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]
