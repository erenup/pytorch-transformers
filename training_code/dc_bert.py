from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertLayer, BertPooler, BertEncoder, BertConfig
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import copy

class BertPoolerQD(nn.Module):
    def __init__(self, config):
        super(BertPoolerQD, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, question_size):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        doc_cls_tensor = hidden_states[:, question_size]
        pooled_output = torch.cat((first_token_tensor, doc_cls_tensor), dim=-1)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class DCBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(DCBERT, self).__init__(config)
        self.num_labels = config.num_labels
        # config.output_hidden_states = True
        bert_later_dropout = 0.2
        self.dropout = nn.Dropout(bert_later_dropout)
        self.copy_from_bert_layer_num = 11
        self.bert = BertModel(config)
        self.bert_position_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.bert_type_id_emb = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.bert_layer = BertLayer(config)
        self.bert_pooler_qd = BertPoolerQD(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        self.init_top_layer_from_bert()

    def init_top_layer_from_bert(self):
            # directly load from bert
        copy_dict = copy.deepcopy(self.bert.encoder.layer[self.copy_from_bert_layer_num].state_dict())
        self.bert_layer.load_state_dict(copy_dict)
        copy_dict = copy.deepcopy(self.bert.embeddings.position_embeddings.state_dict())
        self.bert_position_emb.load_state_dict(copy_dict)
        copy_dict = copy.deepcopy(self.bert.embeddings.token_type_embeddings.state_dict())
        self.bert_type_id_emb.load_state_dict(copy_dict)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                input_ids_a=None, token_type_ids_a=None, attention_mask_a=None,
                input_ids_b=None, token_type_ids_b=None, attention_mask_b=None):

        # outputs_original = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, head_mask=head_mask)
        # pooled_output = self.dropout(pooled_output)
        outputs_a = self.bert(input_ids_a, position_ids=None, token_type_ids=token_type_ids_a, attention_mask=attention_mask_a)
        outputs_b = self.bert(input_ids_b, position_ids=None, token_type_ids=token_type_ids_b, attention_mask=attention_mask_b)

        input_ids = torch.cat((input_ids_a, input_ids_b), dim=1)
        bert_embeddings_a = outputs_a[0]
        bert_embeddings_b = outputs_b[0]
        embeddings_cat = torch.cat((bert_embeddings_a, bert_embeddings_b), dim=1)
        attention_mask = torch.cat((attention_mask_a, attention_mask_b), dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        token_type_ids_a = torch.zeros_like(token_type_ids_a)
        token_type_ids_b = torch.ones_like(token_type_ids_b)
        token_type_ids = torch.cat((token_type_ids_a, token_type_ids_b), dim=1)
        token_type_ids_emb = self.bert_type_id_emb(token_type_ids)
        seq_length = embeddings_cat.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids_a.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embeddings_cat_position_emb = self.bert_position_emb(position_ids)
        transformer_input = embeddings_cat + embeddings_cat_position_emb + token_type_ids_emb
        transformer_outputs = self.bert_layer(transformer_input, extended_attention_mask)
        pooled_output = self.bert_pooler_qd(transformer_outputs[0], question_size=input_ids_a.size(1))

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs_a[2:] + outputs_b[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)