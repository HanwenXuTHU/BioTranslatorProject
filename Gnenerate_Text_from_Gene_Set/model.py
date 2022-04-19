import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class KNNAggregation(nn.Module):

    def __init__(self, k=4, d_model=768, h=1, d_a=8, tau=0.1):
        super(KNNAggregation, self).__init__()
        self.k = k
        self.tau = tau
        self.a_attention = MultiHeadedAttention(h=h, d_model=d_model)
        self.linear_a = nn.Linear(d_model, d_a)
        self.sc_attention = MultiHeadedAttention(h=h, d_model=d_a)
        self.linear_sc = nn.Linear(d_a, 1)

    def forward(self, x_list, dist=None):
        a_list = []
        aggregation = []
        for i in range(self.k):
            a_i = self.a_attention(x_list[i], x_list[i], x_list[i])[:, 0, :]
            a_list.append(self.linear_a(a_i).unsqueeze(1))
        a = torch.cat(a_list, dim=1)
        sc = self.sc_attention(a, a, a)
        w = F.softmax(self.linear_sc(sc) * self.tau, dim=1).unsqueeze(-1)
        if dist != None:
            dist_tensor = []
            for i in range(dist.size(0)):
                dist_tensor.append(dist[i, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
            dist_tensor = torch.cat(dist_tensor, dim=0)
            w = torch.mul(w, dist_tensor)
        for i in range(self.k):
            aggregation.append(x_list[i].unsqueeze(1))
        aggregation = torch.cat(aggregation, dim=1)
        aggregation = torch.sum(w * aggregation, dim=1)
        return aggregation


class Textomics(nn.Module):

    def __init__(self, opt):
        super(Textomics, self).__init__()
        self.opt = opt
        self.aggregation = KNNAggregation(k=opt.K, h=opt.h, d_a=opt.d_a, tau=opt.tau)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_path)
        self.model = AutoModelWithLMHead.from_pretrained(opt.model_path)
        if self.opt.cl:
            self.src_proj, self.tgt_proj = nn.Linear(768, self.opt.cl_d), nn.Linear(768, self.opt.cl_d)
            self.cl_loss = torch.nn.CrossEntropyLoss()

    def forward(self, src_text, tgt_text, dist=None):
        input_emb_list = []
        for k in range(self.opt.K):
            src_k = list(src_text[k])
            src_k_token = self.tokenizer.batch_encode_plus(src_k, max_length=self.opt.max_length,
                                                      add_special_tokens=True,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors='pt')
            input_ids = src_k_token['input_ids']
            input_emb = self.model.get_input_embeddings()(input_ids.to(self.opt.device_0))
            input_emb_list.append(input_emb)
        agg_emb = self.aggregation(input_emb_list, dist=dist.float().to(self.opt.device_0))
        tgt_token = self.tokenizer.batch_encode_plus(tgt_text, max_length=self.opt.max_length,
                                                add_special_tokens=True,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors='pt')
        tgt_ids, tgt_masks = tgt_token['input_ids'], tgt_token['attention_mask']
        lm_labels = tgt_ids.clone()
        lm_labels[tgt_ids[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.model(inputs_embeds=agg_emb.to(self.opt.device_0),
                            decoder_attention_mask=tgt_masks.to(self.opt.device_0),
                            labels=lm_labels.to(self.opt.device_0))
        loss = outputs.loss
        if self.opt.cl:
            tgt_emb = self.model.get_input_embeddings()(tgt_ids.to(self.opt.device_0))
            tgt_encoding = self.model.get_encoder()(inputs_embeds=tgt_emb.to(self.opt.device_0)).last_hidden_state
            src_encoding = self.model.get_encoder()(inputs_embeds=agg_emb.to(self.opt.device_0)).last_hidden_state
            tgt_proj = self.tgt_proj(tgt_encoding[:, 0, :])
            src_proj = self.src_proj(src_encoding[:, 0, :])
            tgt_proj = torch.nn.functional.normalize(tgt_proj, p=2, dim=1)
            src_proj = torch.nn.functional.normalize(src_proj, p=2, dim=1)
            logits = torch.mm(src_proj, tgt_proj.permute(1, 0))
            labels = torch.arange(src_proj.size(0)).to(self.opt.device_0)
            loss_src = self.cl_loss(logits, labels)
            loss_tgt = self.cl_loss(logits.permute(1, 0), labels)
            loss = outputs.loss + self.opt.cl_beta * (loss_src + loss_tgt) / 2

        return outputs, loss

    def generate(self, src_text, dist=None):
        input_emb_list = []
        for k in range(self.opt.K):
            src_k = list(src_text[k])
            src_k_token = self.tokenizer.batch_encode_plus(src_k, max_length=self.opt.max_length,
                                                           add_special_tokens=True,
                                                           padding='max_length',
                                                           truncation=True,
                                                           return_tensors='pt')
            input_ids = src_k_token['input_ids']
            input_emb = self.model.get_input_embeddings()(input_ids.to(self.opt.device_0))
            input_emb_list.append(input_emb)
        agg_emb = self.aggregation(input_emb_list, dist=dist.to(self.opt.device_0))
        encoder_outputs = self.model.get_encoder()(inputs_embeds=agg_emb.to(self.opt.device_0))
        pred_ids = self.model.generate(
                        encoder_outputs=encoder_outputs,
                        max_length=self.opt.max_length,
                        num_beams=self.opt.num_beams,
                        repetition_penalty=self.opt.repetition_penalty,
                        length_penalty=self.opt.length_penalty,
                        early_stopping=True)
        preds = [self.tokenizer.decode(g, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True) for g in pred_ids]

        return preds



