from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import config

from data_util import config # from this codebase

use_cuda = config.use_gpu and torch.cuda.is_available() # check system

random.seed(123) #what for? Reprodcibility
torch.manual_seed(123)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(123)

# init lstm weights to some nonzero values (zero means it forgets prev memory)
def init_lstm_wt(lstm):
	for names in lstm._all_weights:
		for name in names:
			if name.startswith('weight_'):
				wt = getattr(lstm, name)
				wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag) # what is data.unif
			elif name.startswith('bias_'):
				# set forget bias to 1
				bias = getattr(lstm,name)
				n = bias.size(0)
				start, end = n//4, n//2
				bias.data.fill_(0.)
				bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
	linear.weight.data.normal(std=config.trunc_norm_init_std)
	if linear.bias is not None:
		linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
	wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
	wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.embedding = nn.Embedding(confi.vocab_size, config.emb_dim)
		init_wt_normal(self.embedding.weight) # normalize embedding weights

		self.lstm = nn.LSTM(config.emb_dim,config.hidden_dim,num_layers=1,batch_first=True,bidirectional)
		init_lstm_wt(self.lstm)

		self.W_h = nn.Linear(config.hidden_dim *2, config.hidden_dim *2, bias = False)


	def forward(self, input, seq_lens):
		embedded = self.embedding(input)

		packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
		output, hidden = self.lstm(packed)

		encoder_outputs, _ = pad_packed_sequence(output, batch_first=True) # h dim = B x t_k x n
		encoder_outputs = encoder_outputs.contiguous()

		encoder_feature = encoder_outputs.view(-1, 2* config.hidden_dim) # B * t_k x 2*hidden_dim
		encoder_feature = self.W_h(encoder_feature)

		return encoder_outputs, encoder_feature, hidden

# not sure what this is for
class ReduceState(nn.Module):
	def __init__(self):
		super(ReduceState, self).__init__()

		self.reduce_h = nn.Linear(config.hidden_dim *2, config.hidden_dim)
		init_linear_wt(self.reduce_h)

		self.reduce_c = nn.Linear(config.hidden_dim *2 , config.hidden_dim)
		init_linear_wt(self.reduce_c)

	def forward(self,hidden):
		h, c = hidden # h, c dim = 2 x b x hidden_dim
		h_in = h.transpose(0,1).contiguous().view(-1,config.hidden_dim*2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		# attention
		if config.is_coverage:
			self.W_c = nn.Linear(1, config.hidden_dim * 2, bias = False)

		self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim*2)
		self.v = nn.Linear(config.hidden_dim *2,1,bias=False)


    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.attention_network = Attention()
		# decoder 
		self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

		init_wt_normal(self.embedding.weight)

		self.x_context = nn.Linear(config.hidden_dim *2 + config.emb_dim, config.emb_dim)

		        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p+vocab
        self.out1 = nn.Linear(config.hidden_dim*3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
    	c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

    	if not self.training and step == 0:
    		h_decoder, c_decoder = s_t_1
    		s_t_hat = torch.cat((h_decoder.view(-1,config.hidden_dim),
    			c_decoder.view(-1, config.hidden_dim)), 1) # B x 2*hidden_dim

    		c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask,
    			coverage)

    		coverage = coverage_next

    	y_t_1_embed = self.embedding(y_t_1)