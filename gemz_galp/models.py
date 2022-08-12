"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import sys
from functools import partial

import gemz.models as gm

from galp.graph import StepSet

export = StepSet()

# Stpes. Wrapping could be automated, but would not be much easier to maintain
# and would break many static analysis programs

fit = export(gm.fit)
predict_loo = export(gm.predict_loo)
eval_loss = export(gm.eval_loss)
fold = export(gm.fold, items=2) # -> (train, test)
aggregate_losses = export(gm.aggregate_losses)

# Meta steps:

_self = sys.modules[__name__]

fit_eval = partial(gm.fit_eval, ops=_self)
cv_fit_eval = partial(gm.cv_fit_eval, ops=_self)
