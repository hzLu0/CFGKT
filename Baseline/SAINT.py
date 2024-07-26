import torch
from torch import nn
import copy
from torch.nn.init import xavier_uniform_,constant_


class SAINT+(nn.Module):
    def __init__(self, n_encoder, n_decoder, enc_heads, dec_heads, n_dims, total_ex, total_cat, total_responses,
                 seq_len):
        super(SAINT, self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.seq_len = seq_len
        self.enocder = nn.ModuleList(
            [copy.deepcopy(EncoderBlock(enc_heads, n_dims)) for i in range(n_encoder)]).to(device)
        self.decoder = nn.ModuleList(
            [copy.deepcopy(DecoderBlock(dec_heads, n_dims, total_responses, self.seq_len)) for j in range(n_decoder)]).to(device)
        self.fc = nn.Linear(n_dims, 1)
        self.exercise_embed = nn.Embedding(total_ex+1, n_dims, padding_idx=total_ex).to(device)
        self.category_embed = nn.Embedding(total_cat+1, n_dims, padding_idx=total_cat).to(device)
        self.response_embed = nn.Embedding(total_responses*2+2, n_dims,padding_idx=total_responses*2+1).to(device)
        self.pos_embedding = nn.Embedding(self.seq_len, n_dims).to(device)
        self.timelag_embed = nn.Embedding(1442,n_dims,padding_idx=1441).to(device)
        self.timespend_embed = nn.Embedding(302,n_dims,padding_idx=301).to(device)
        self.lagtime_map = nn.Linear(n_dims*3,n_dims).to(device)
        self.encoder_map = nn.Linear(n_dims*4,n_dims*2).to(device)
        self.encodermap = nn.Linear(n_dims*2,n_dims).to(device)
        self._reset_parameters()
        # self.map = nn.Linear(n_dims*2,n_dims).to(device)
    def _reset_parameters(self):
        xavier_uniform_(self.lagtime_map.weight)
        xavier_uniform_(self.encoder_map.weight)
        xavier_uniform_(self.encodermap.weight)
        constant_(self.lagtime_map.bias, 0.)
        constant_(self.encoder_map.bias, 0.)
        constant_(self.encodermap.bias, 0.)

    def forward(self, in_exercise, in_category, in_response, dotime, lagtime):
        in_exercise = self.exercise_embed(in_exercise)
        in_category = self.category_embed(in_category)
        in_response = self.response_embed(in_response)
        in_dotime = self.timespend_embed(dotime)
        in_lagtime = self.timelag_embed(lagtime).to(device)
        pos_id = torch.arange(self.seq_len).unsqueeze(0).to(device)
        pos_embed = self.pos_embedding(pos_id).to(device)
        q_out = self.encoder_map(torch.cat((in_category,in_exercise,in_dotime,in_lagtime),dim=-1)).to(device)
        q_out = self.encodermap(q_out)+pos_embed
        # q_out = pos_embed + in_category + in_exercise
        # qa_out = in_response+pos_embed+in_dotime+in_lagtime
        qa_out = self.lagtime_map(torch.cat((in_response,in_dotime,in_lagtime),dim=-1))+pos_embed
        for i,layer in enumerate(self.enocder):
            q_out = layer(q_out).to(device)
        for i,layer in enumerate(self.decoder):
            qa_out = layer(qa_out,q_out).to(device)
        return torch.sigmoid(self.fc(qa_out))

class EncoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims):
        super(EncoderBlock, self).__init__()
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=n_dims)

    def forward(self, input_e):
        output = self.multihead(q_input=input_e, kv_input=input_e)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, total_responses, seq_len):
        super(DecoderBlock, self).__init__()
        self.seq_len = seq_len
        self.layer_norm = nn.LayerNorm(n_dims).to(device)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                                         num_heads=n_heads,
                                                         dropout=0.2).to(device)
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=n_dims).to(device)

    def forward(self, input_r, encoder_output):
        out = input_r.permute(1, 0, 2).to(device)

        mask = ut_mask(out.size(0)).to(device)
        out_atten, weights_attent = self.multihead_attention(query=out,
                                                             key=out,
                                                             value=out,
                                                             attn_mask=mask)
        out_atten = out_atten.permute(1,0,2)
        norm_outattn = self.layer_norm(out_atten + input_r)
        output = self.multihead(q_input=encoder_output, kv_input=norm_outattn)
        return output

class FFN(nn.Module):
    def __init__(self, features):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(features, features)
        self.layer2 = nn.Linear(features, features)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out = self.drop(self.relu(self.layer1(x)))
        out = self.layer2(out)
        return out

class MultiHeadWithFFN(nn.Module):
    def __init__(self, n_heads, n_dims, dropout=0.2):
        super(MultiHeadWithFFN, self).__init__()
        self.n_dims = n_dims
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                                         num_heads=n_heads,
                                                         dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(n_dims)
        self.ffn = FFN(features=n_dims)
        self.layer_norm2 = nn.LayerNorm(n_dims)

    def forward(self, q_input, kv_input):
        q_input = q_input.permute(1, 0, 2).to(device)
        kv_input = kv_input.permute(1, 0, 2).to(device)
        mask = ut_mask(q_input.size(0)).to(device)
        out_atten, weights_attent = self.multihead_attention(query=q_input,
                                                             key=kv_input,
                                                             value=kv_input,
                                                             attn_mask=mask)
        out_atten = self.layer_norm1(out_atten + q_input).permute(1, 0, 2)
        output = self.ffn(out_atten)
        output_norm = self.layer_norm2(output + out_atten)
        return output_norm

def ut_mask(seq_len):
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)

