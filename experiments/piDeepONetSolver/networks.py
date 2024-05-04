import torch
import torch.nn as nn
import numpy as np


class PIDeepONet(nn.Module):
    def __init__(self, n_field_dim,  n_branch_in, n_trunk_in, n_out,
        num_hidden_layers, hidden_features, nonlinearity) -> None:
        super().__init__()

        self.branch_net = MLP(n_branch_in, n_out, num_hidden_layers,
            hidden_features, nonlinearity=nonlinearity)
        self.trunk_net = MLP(n_trunk_in, n_out, num_hidden_layers,
            hidden_features, nonlinearity=nonlinearity)

        self.b = nn.Parameter(torch.zeros(1, n_field_dim))

        self.n_field_dim = n_field_dim
        assert n_out % n_field_dim == 0
        self.split_size = n_out // n_field_dim

    def forward(self, u0, x, t):
        B = self.branch_net(u0) # (n_out, )
        T = self.trunk_net(torch.cat([x, t], dim=-1)) # (batch_size, n_out)
        batch_size = T.shape[0]
        B = B.unsqueeze(0).view(1, self.n_field_dim, -1)
        T = T.view(batch_size, self.n_field_dim, -1)
        out = torch.sum(B * T, dim=-1) + self.b # (batch_size, n_field_dim)
        return out.squeeze(-1)


############################### SIREN ################################
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None),
                         'tanh': (nn.Tanh(), None, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nl])

        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])

        self.net.append(nn.Linear(hidden_features, out_features))
        if not outermost_linear:
            self.net.append(nl)

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, weights=None):
        output = self.net(coords)
        if weights is not None:
            output = output * weights
        return output


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))
