from .model_split import NeuralFluidSplit
# from .model_all import NeuralFluidAll
# from .model_auxbound import NeuralFluidAuxbound
# from .model_split_auxbound import NeuralFluidSplitAuxbound


def get_model(cfg):
    if cfg.mode == 'split':
        return NeuralFluidSplit(cfg)
    # elif cfg.mode == 'all': # NOTE: re-implement later
    #     return NeuralFluidAll(cfg)
    # elif cfg.mode == 'auxbound':  # NOTE: re-implement later, merge with 'all'
    #     return NeuralFluidAuxbound(cfg)
    # elif cfg.mode == 'split_auxbound': # NOTE: buggy! pressure solve boundary not right
    #     raise NotImplementedError
    #     # return NeuralFluidSplitAuxbound(cfg)
    else:
        raise NotImplementedError
