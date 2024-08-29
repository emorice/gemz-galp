"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import sys

from gemz.models import ops as _ops

from galp import step, query
import galp.task_types

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

@step
def s_extract_cv_losses(grid):
    """
    Wrap the extraction, guard against task refs
    """
    if isinstance(grid, galp.task_types.TaskRef):
        return s_extract_cv_losses(query(grid, '$base'))
    return _ops.s_extract_cv_losses(grid)

# Meta steps:

_self = sys.modules[__name__]

def extract_cv(grid):
    """
    Gather only the cross-validation data necessary for decision and plotting

    (The sum of all the cross validation models is commonly too large for memory)

    Args:
        t_grid: the grid task
    """
    # Give only the base grid as the full grid precisely is too large
    return s_extract_cv_losses(query(grid, '$base'))

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
