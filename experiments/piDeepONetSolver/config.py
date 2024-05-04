import os
import argparse
import json
import shutil
from utils.file_utils import ensure_dirs


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase = 'train'):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.results_dir = os.path.join(self.exp_dir, 'results')

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # load saved config if not training
        if not self.is_train:
            print(self.exp_dir)
            assert os.path.exists(self.exp_dir)
            config_path = os.path.join(self.exp_dir, 'config.json')
            print(f"Load saved config from {config_path}")
            with open(config_path, 'r') as f:
                saved_args = json.load(f)
            for k, v in saved_args.items():
                if not hasattr(self, k):
                    self.__setattr__(k, v)
            return
        else:
            if os.path.exists(self.exp_dir):
                response = input('Experiment log/model already exists, overwrite? (y/n) ')
                if response != 'y':
                    exit()
                shutil.rmtree(self.exp_dir)
            ensure_dirs([self.log_dir, self.model_dir, self.results_dir])

        # save this configuration for backup
        backup_dir = os.path.join(self.exp_dir, "backup")
        ensure_dirs(backup_dir)
        os.system(f"cp *.py {backup_dir}/")
        # os.system(f"mkdir {backup_dir}/models | cp models/*.py {backup_dir}/models/")
        # os.system(f"mkdir {backup_dir}/utils | cp utils/*.py {backup_dir}/utils/")
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        if self.is_train:
            # model configuration
            self._add_network_config_(parser)

            # training or testing configuration
            self._add_training_config_(parser)
        else:
            self._add_testing_config_(parser)

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="checkpoints", 
            help="path to project folder where models and logs will be saved")
        group.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=0, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument('--network', type=str, default='siren', choices=['siren', 'grid'])
        group.add_argument('--num_hidden_layers', type=int, default=3)
        group.add_argument('--hidden_features', type=int, default=256)
        group.add_argument('--nonlinearity',type=str, default='sine')

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        # group.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default=None, required=False, help="checkpoint at x timestep to restore")
        # group.add_argument('--ckpt_timestep', type=int, default=0, required=False, help="desired checkpoint timestep to restore")       
        # group.add_argument('--save_frequency', type=int, default=1000, help="save models every x steps")
        group.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
        group.add_argument('--max_n_iters', type=int, default=50000, help='number of epochs to train per scale')
        # group.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
        # group.add_argument('--lr_stepsize', type=int, help='scheduler lr_stepsize', default=10000)
        group.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0005')
        group.add_argument('--grad_clip', type=float, default=-1, help='grad clipping, l2 norm')
        group.add_argument('--early_stop', action='store_true', help="early_stopping")
        # group.add_argument('--dim', type=int, default=2, help='dimension of the fluid simulation')
        
        group.add_argument('--dt', type=float, default=0.001)
        group.add_argument('-T','--n_timesteps', type=int, default=50)
        group.add_argument('--visc', type=float, default=0)
        group.add_argument('--diff', type=float, default=0)
        group.add_argument('-sr', '--sample_resolution', type=int, default=128)
        group.add_argument('-vr', '--vis_resolution', type=int, default=32)
        group.add_argument('--fps', type=int, default=10)

        group.add_argument('--src', type=str, default=None, help='which example to use', required=True)
        group.add_argument('--src_duration', type=int, default=1, help='source duration')
        group.add_argument('--src_start_frame', type=int, default=1, help='starting frame of the source loaded')
        # group.add_argument('--stage', type=str, default=None, choices=['add_source', 'step_velocity', 'solve_pressure'], 
        #     required=True)
        group.add_argument('--boundary_cond', type=str, default='zero', choices=['zero', 'none'])
        # group.add_argument('--incompressible', action='store_true', help="use pressure")
        group.add_argument('--grad_sup', action='store_true', help="supervise gradient when adding source")
        group.add_argument('--save_h5', action='store_true', help="save grid values as h5 file")
        # group.add_argument('--init_p', action='store_true', help="init p with gt")
        group.add_argument('--no_dudt', action='store_true', help="remove dudt from the N-S loss")
        group.add_argument('-m', '--mode', type=str, default='split', choices=['split', 'all', 'auxbound', 'split_auxbound'], 
            help="operator splitting or solve in one loss")
        group.add_argument('-ti', '--time_integration', type=str, default='semi_lag', choices=['implicit', 'semi_lag'],
            help="time integration method")
        group.add_argument('--alpha', type=float, default=0.5, help="blending weight for implicit and explicit time integration")
        # group.add_argument('--gt_taylorgreen_p', action='store_true', help="use ground truth taylor green pressure")
        group.add_argument('--sample', type=str, default='random', choices=['random', 'uniform', 'random+uniform', 'fixed'],
                            help='The sampling strategy to be used during the training.')

        group.add_argument('--debug', action='store_true', help="debug mode, save more intermediate results")

        group.add_argument('--use_disc_p', action='store_true', help="use discrete pressure solve")

        group.add_argument('--use_density', action='store_true', help="also model density field")
        group.add_argument('--t_range', type=float, default=2.5, help='time range')
    
    def _add_testing_config_(self, parser):
        """testing configuration"""
        group = parser.add_argument_group('testing')
        group.add_argument('-vr', '--vis_resolution', type=int, default=32)
        group.add_argument('--fps', type=int, default=10)
