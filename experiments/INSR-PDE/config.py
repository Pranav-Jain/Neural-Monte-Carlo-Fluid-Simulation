import os
import argparse
import json
import shutil


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase='train'):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.tag)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # load saved config if not training
        if not self.is_train:
            if not os.path.exists(self.exp_dir):
                raise RuntimeError(f"Experiment checkpoint {self.exp_dir} not exists.")
            config_path = os.path.join(self.exp_dir, 'config.json')
            print(f"Load saved config from {config_path}")
            with open(config_path, 'r') as f:
                saved_args = json.load(f)
            for k, v in saved_args.items():
                if not hasattr(self, k):
                    self.__setattr__(k, v)
            return

        if args.ckpt is None and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        for path in [self.log_dir, self.model_dir]:
            os.makedirs(path)

        # save this configuration for backup
        backup_dir = os.path.join(self.exp_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        os.system(f"cp *.py {backup_dir}/")
        os.system(f"mkdir -p {backup_dir}/base | cp base/*.py {backup_dir}/base/")
        os.system(f"mkdir -p {backup_dir}/{args.pde} | cp {args.pde}/*.py {backup_dir}/{args.pde}/")

        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parent_parser = argparse.ArgumentParser(add_help=False)
        self._add_basic_config_(parent_parser)
        if self.is_train:
            self._add_network_config_(parent_parser)
            self._add_training_config_(parent_parser)
            self._add_timestep_config_(parent_parser)
        else:
            self._add_recap_config_(parent_parser)
        
        parser = argparse.ArgumentParser(add_help=False)      
        subparsers = parser.add_subparsers(dest="pde", required=True)
        parser_adv = subparsers.add_parser("advection", parents=[parent_parser])
        parser_flu = subparsers.add_parser("fluid", parents=[parent_parser])
        parser_ela = subparsers.add_parser("elasticity", parents=[parent_parser])
        if self.is_train:
            self._add_advection_config_(parser_adv)
            self._add_fluid_config_(parser_flu)
            self._add_elasticity_config_(parser_ela)

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="checkpoints", 
            help="path to project folder where models and logs will be saved")
        group.add_argument('--tag', type=str, default="run", help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=0, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument('--network', type=str, default='siren', choices=['siren', 'grid'])
        group.add_argument('--num_hidden_layers', type=int, default=3)
        group.add_argument('--hidden_features', type=int, default=64)
        group.add_argument('--nonlinearity',type=str, default='sine')

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--ckpt', type=str, default=None, required=False, help="desired checkpoint to restore")
        group.add_argument('--vis_frequency', type=int, default=1000, help="visualize output every x iterations")
        group.add_argument('--max_n_iters', type=int, default=20000, help='number of iterations to train every time step')
        group.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        group.add_argument('-sr', '--sample_resolution', type=int, default=128, help='number of samples per iterations')
        group.add_argument('-vr', '--vis_resolution', type=int, default=500)
        group.add_argument('--early_stop', action=argparse.BooleanOptionalAction, default=True)
    
    def _add_recap_config_(self, parser):
        """recap configuration"""
        group = parser.add_argument_group('recap')
        group.add_argument('-o', '--output', type=str, default="recap", help="output folder under exp folder")
        group.add_argument('-vr', '--vis_resolution', type=int, default=500)
        
    def _add_timestep_config_(self, parser):
        """configuration for pde time stepping"""
        group = parser.add_argument_group('timestep')
        group.add_argument('--init_cond', type=str, default=None, help='which example to use for initial condition')
        group.add_argument('--dt', type=float, default=0.001, help='time step size')
        group.add_argument('-T','--n_timesteps', type=int, default=50, help='number of time steps')
        group.add_argument('--fps', type=int, default=10)

    def _add_advection_config_(self, parser):
        group = parser.add_argument_group('advection')
        group.add_argument('-L','--length', type=float, default=4.0, help='field length')
        group.add_argument('--vel', type=float, default=0.25, help='constant velocity value')

    def _add_fluid_config_(self, parser):
        pass

    def _add_elasticity_config_(self, parser):
        group = parser.add_argument_group('elasticity')
        group.add_argument('--dim', type=int, default=2)
        group.add_argument('--sample_pattern', type=str, nargs='*',
                            default=['random', 'uniform'],
                            help='The sampling strategy to be used during the training.')

        group.add_argument('--energy', type=str, nargs='*', 
                            default=['arap', 'kinematics', 'external', 'constraint'],
                            help='The energy to be used.')
        group.add_argument('--ratio_constraint', type=float, default=1e3)
        group.add_argument('--ratio_volume', type=float, default=1e1)
        group.add_argument('--ratio_arap', type=float, default=1e0)
        group.add_argument('--ratio_collide', type=float, default=1e0)
        group.add_argument('--ratio_kinematics', type=float, default=1e0)

        group.add_argument('--use_mesh', type=bool, default=False)
        group.add_argument('--mesh_path', type=str, default="./elasticity/data/woody.obj", help="path to the mesh")

        group.add_argument('-T_ext', '--external_force_timesteps', type=int, default=5)
        group.add_argument('-f_ext_x', '--external_force_x', type=float, default=0)
        group.add_argument('-f_ext_y', '--external_force_y', type=float, default=0)
        group.add_argument('-f_ext_z', '--external_force_z', type=float, default=0)

        group.add_argument('-fix_right_x', '--constraint_right_offset_x', type=float, default=1e0)
        group.add_argument('-fix_right_y', '--constraint_right_offset_y', type=float, default=0)
        group.add_argument('-fix_right_z', '--constraint_right_offset_z', type=float, default=0)

        group.add_argument('--plane_height', type=float, default=-2)

        group.add_argument('-collide_circle_x', '--collide_circle_x', type=float, default=0)
        group.add_argument('-collide_circle_y', '--collide_circle_y', type=float, default=-2e0)
        group.add_argument('-collide_circle_z', '--collide_circle_z', type=float, default=0)
        group.add_argument('-collide_circle_r', '--collide_circle_radius', type=float, default=1)
