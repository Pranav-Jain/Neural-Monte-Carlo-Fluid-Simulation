import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_network(cfg, in_features, out_features):
    if cfg.network == 'siren':
        return MLP(in_features, out_features, cfg.num_hidden_layers,
            cfg.hidden_features, nonlinearity=cfg.nonlinearity)
    elif cfg.network == 'grid':
        return GridImplicit(in_features, out_features, cfg.num_hidden_layers,
            cfg.hidden_features, cfg.n_levels, cfg.fdim, cfg.fsize, cfg.nonlinearity)
    else:
        raise NotImplementedError


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
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

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

    def forward(self, x, t):
        coords = torch.cat([x, t], dim=-1)
        output = self.net(coords)
        return output



############################### Gird-based implicit ################################
class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fm = nn.Parameter(torch.randn(1, fdim, 1, 1).expand(1, fdim, fsize + 1, fsize + 1) * 0.01) # spatially equal initialization

    def forward(self, x):
        if len(x.shape) == 2: # (N, 2)
            sample_coords = x.reshape(1, x.shape[0], 1, -1) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=False, padding_mode='zeros')[0,:,:,0].transpose(0, 1)
        else: # grid points (H, W, 2)
            sample = F.grid_sample(self.fm, x.unsqueeze(0), 
                                   align_corners=False, padding_mode='zeros')[0].permute(1, 2, 0)
        
        return sample


def make_mlp(in_features, out_features, hidden_features, num_hidden_layers, act='relu'):
    nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                        'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                        'elu':(nn.ELU(inplace=True), init_weights_elu, None)}
    nl, nl_weight_init, first_layer_init = nls_and_inits[act]
    layer_list = [nn.Linear(in_features, hidden_features), nl]
    for i in range(num_hidden_layers):
        layer_list.extend([nn.Linear(hidden_features, hidden_features), nl])
    layer_list.extend([nn.Linear(hidden_features, out_features)])
    net = nn.Sequential(*layer_list)
    if nl_weight_init is not None:
        net.apply(nl_weight_init)
    if first_layer_init is not None:
        net[0].apply(first_layer_init)
    return net


class GridImplicit(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers=1, hidden_features=128, 
            n_levels=5, fdim=32, fsize=4, nonlinearity='relu'):
        super().__init__()

        self.fdim = fdim
        self.fsize = fsize
        self.n_levels = n_levels

        self.features = nn.ModuleList([])
        s = fsize
        for i in range(self.n_levels):
            self.features.append(FeatureVolume(self.fdim, s))
            s = s * 2
        print("feature sizes:", [self.features[i].fm.shape for i in range(len(self.features))])

        self.input_dim = self.fdim + in_features
        self.num_decoder = self.n_levels 

        self.louts = nn.ModuleList([])
        for i in range(self.num_decoder):
            self.louts.append(
                make_mlp(self.input_dim, out_features, hidden_features, num_hidden_layers, nonlinearity)
            )
    
    def get_features(self):
        return [self.features[i].fm for i in range(len(self.features))]

    def forward(self, x, return_lst=True):
        # Query
        l = []
        samples = []

        for i in range(self.n_levels):
            # Query features
            sample = self.features[i](x)
            samples.append(sample)
            
            # Sum queried features
            if i > 0:
                samples[i] += samples[i-1]
            
            # Concatenate xyz
            ex_sample = samples[i]
            ex_sample = torch.cat([x, ex_sample], dim=-1)

            d = self.louts[i](ex_sample)

            l.append(d)
        # if self.training:
        #     self.loss_preds = l

        if return_lst:
            return l
        else:
            return l[-1]



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
