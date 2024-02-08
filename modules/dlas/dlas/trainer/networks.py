import importlib
import logging
import os
import pkgutil
import sys
from inspect import getmembers, isfunction, signature

logger = logging.getLogger('base')


class RegisteredModelNameError(Exception):
    def __init__(self, name_error):
        super().__init__(
            f'Registered DLAS modules must start with `register_`. Incorrect registration: {name_error}')


# Decorator that allows API clients to show DLAS how to build a nn.Module from an opt dict.
# Functions with this decorator should have a specific naming format:
# `register_<name>` where <name> is the name that will be used in configuration files to reference this model.
# Functions with this decorator are expected to take a single argument:
# - opt: A dict with the configuration options for building the module.
# They should return:
# - A torch.nn.Module object for the model being defined.
def register_model(func):
    if func.__name__.startswith("register_"):
        func._dlas_model_name = func.__name__[9:]
        assert func._dlas_model_name
    else:
        raise RegisteredModelNameError(func.__name__)
    func._dlas_registered_model = True
    return func

# this had some weird kludge that I don't understand needing to have a reference frame around the current working directory
# it works better when you set it relative to this file instead
# however, this has very different behavior when importing DLAS from outside the repo, rather than spawning a shell instance to a script within it
# I can't be assed to deal with that headache at the moment, I just want something to work right now without needing to touch a shell

# inject.py has a similar loader scheme, be sure to mirror it if you touch this too


def find_registered_model_fns(base_path='models'):
    found_fns = {}
    path = os.path.normpath(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), f'../{base_path}'))

    module_iter = pkgutil.walk_packages([path])
    for mod in module_iter:
        if mod.ispkg:
            EXCLUSION_LIST = ['flownet2']
            if mod.name not in EXCLUSION_LIST:
                found_fns.update(find_registered_model_fns(
                    f'{base_path}/{mod.name}'))
        else:
            mod_name = f'dlas/{base_path}/{mod.name}'.replace('/', '.')
            importlib.import_module(mod_name)
            for mod_fn in getmembers(sys.modules[mod_name], isfunction):
                if hasattr(mod_fn[1], "_dlas_registered_model"):
                    found_fns[mod_fn[1]._dlas_model_name] = mod_fn[1]
    return found_fns


class CreateModelError(Exception):
    def __init__(self, name, available):
        super().__init__(f'Could not find the specified model name: {name}. Tip: If your model is in a'
                         f' subdirectory, that directory must contain an __init__.py to be scanned. Available models:'
                         f'{available}')


def create_model(opt, opt_net, other_nets=None):
    which_model = opt_net['which_model']
    # For backwards compatibility.
    if not which_model:
        which_model = opt_net['which_model_G']
    if not which_model:
        which_model = opt_net['which_model_D']
    registered_fns = find_registered_model_fns()
    if which_model not in registered_fns.keys():
        raise CreateModelError(which_model, list(registered_fns.keys()))
    num_params = len(signature(registered_fns[which_model]).parameters)
    if num_params == 2:
        return registered_fns[which_model](opt_net, opt)
    else:
        return registered_fns[which_model](opt_net, opt, other_nets)
