"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import sys
from functools import wraps

from gemz.models import ops as _ops

from galp.graph import Block

export = Block()

# Steps. Wrapping could be automated, but would not be much easier to maintain
# and would break many static analysis programs

predict_loo = export(_ops.predict_loo)
eval_loss = export(_ops.eval_loss)
metric = export(_ops.metric)
fold = export(_ops.fold, items=3) # -> (train, test, mask)
aggregate_losses = export(_ops.aggregate_losses)
aggregate_residuals = export(_ops.aggregate_residuals)
select_best = export(_ops.select_best)

# Meta steps:

_self = sys.modules[__name__]

# pylint: disable=missing-function-docstring
# @wraps sets the docstring but pylint does not seem to know that

@export
@wraps(_ops.fit)
def fit(*args, **kwargs):
    return _ops.fit(*args, **kwargs, _ops=_self)

@export
@wraps(_ops.fit_eval)
def fit_eval(*args, **kwargs):
    return _ops.fit_eval(*args, **kwargs, _ops=_self)

@wraps(_ops.cv_fit_eval)
def cv_fit_eval(*args, **kwargs):
    return _ops.cv_fit_eval(*args, **kwargs, _ops=_self)

@export
@wraps(_ops.cv_residualize)
def cv_residualize(*args, **kwargs):
    return _ops.cv_residualize(*args, **kwargs, _ops=_self)

@wraps(_ops.build_eval_grid)
def build_eval_grid(*args, **kwargs):
    return _ops.build_eval_grid(*args, **kwargs, _ops=_self)

class Model(_ops.Model):
    """
    Model subclass that makes the galp-aware ops module accessible to methods
    """
    ops = _self
