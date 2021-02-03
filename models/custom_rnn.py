import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import torch.nn.functional as F


class GenericMGU(nn.Module):
    def __init__(self, T_no, H_no, layer_no, in_no, device):
        super().__init__()
        
        self.T_no = T_no
        self.H_no = H_no
        self.layer_no = layer_no
        self.in_no = in_no
        self.device = device
        
        self.rnn = StackedCustom(layer_no, CustomLayer,
                      first_layer_args=[MGUCell, in_no, H_no],
                      other_layer_args=[MGUCell, H_no, H_no])
        
        self.linear = nn.Linear(self.H_no, 1)
        
    def forward(self, S):
        T_data = S.shape[0]
        
        S = S.unsqueeze(1)
        states = [CustomState(torch.randn(1, self.H_no).to(self.device))
                  for _ in range(self.layer_no)]
        rnn_out, out_state = self.rnn(S, states)
        out = self.linear(rnn_out.squeeze(1)).flatten()
        
        return out
    
CustomState = namedtuple('CustomState', 'hx')

class MGUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(MGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(2 * hidden_size))
        self.bias_hh = Parameter(torch.randn(2 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor, Tuple[Tensor]]:

        gi = F.linear(input.view(1, -1), self.weight_ih, self.bias_ih)
        gh = F.linear(state[0], self.weight_hh, self.bias_hh)

        i_f, i_n = gi.chunk(2, 1)
        h_f, h_n = gh.chunk(2, 1)

        forgetgate = torch.sigmoid(i_f + h_f)
        newgate = torch.tanh(i_n + forgetgate * h_n)
        hy = newgate + (1 - forgetgate) * (state[0] - newgate)
        out = hy

        return out, (hy,)

class CustomLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(CustomLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor, Tuple[Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedCustom(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedCustom, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor]]]:
        # List[CustomState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states

