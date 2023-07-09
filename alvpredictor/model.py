from collections import OrderedDict

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ALVPredictor(nn.Module):
    def __init__(self, model, num_class, hidden_size=768):
        super().__init__()

        self.encoder = model.encoder
        self.alv_predictor = nn.Linear(hidden_size, num_class)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, phonemes, src_masks=None):
        output = self.encoder(phonemes, attention_mask=(~src_masks).squeeze(-1).int())
        alv_pred = self.alv_predictor(self.dropout(output.last_hidden_state))
        if src_masks is not None:
            alv_pred = alv_pred.masked_fill(src_masks, 0.0)
        
        return alv_pred
