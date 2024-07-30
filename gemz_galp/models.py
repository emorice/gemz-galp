"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import sys

from gemz.models import ops as _ops

from galp import step

# pylint: disable=missing-function-docstring
# @wraps sets the docstring but pylint does not seem to know that

# Steps. Wrapping could be automated, but would not be much easier to maintain
# and would break many static analysis programs

@step
def predict_loo(*args, **kwargs):
    return _ops.predict_loo(*args, **kwargs)

@step
def eval_loss(*args, **kwargs):
    return _ops.eval_loss(*args, **kwargs)

@step
def metric(*args, **kwargs):
    return _ops.metric(*args, **kwargs)

@step(items=3) # -> (train, test, mask)
def fold(*args, **kwargs):
    return _ops.fold(*args, **kwargs)

@step
def aggregate_losses(*args, **kwargs):
    return _ops.aggregate_losses(*args, **kwargs)

@step
def aggregate_residuals(*args, **kwargs):
    return _ops.aggregate_residuals(*args, **kwargs)

@step
def select_best(*args, **kwargs):
    return _ops.select_best(*args, **kwargs)


# Meta steps:

_self = sys.modules[__name__]

@step
def fit(*args, **kwargs):
    return _ops.fit(*args, **kwargs, _ops=_self)

@step
def fit_eval(*args, **kwargs):
    return _ops.fit_eval(*args, **kwargs, _ops=_self)

def cv_fit_eval(*args, **kwargs):
    return _ops.cv_fit_eval(*args, **kwargs, _ops=_self)

@step
def cv_residualize(*args, **kwargs):
    return _ops.cv_residualize(*args, **kwargs, _ops=_self)

def build_eval_grid(*args, **kwargs):
    return _ops.build_eval_grid(*args, **kwargs, _ops=_self)

class Model(_ops.Model):
    """
    Model subclass that makes the galp-aware ops module accessible to methods
    """
    ops = _self
