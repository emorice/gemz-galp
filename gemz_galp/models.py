"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import sys
from functools import partial

from gemz.models import ops

from galp.graph import StepSet

export = StepSet()

# Steps. Wrapping could be automated, but would not be much easier to maintain
# and would break many static analysis programs

_fit = export(ops.fit)
predict_loo = export(ops.predict_loo)
eval_loss = export(ops.eval_loss)
fold = export(ops.fold, items=3) # -> (train, test, mask)
aggregate_losses = export(ops.aggregate_losses)
aggregate_residuals = export(ops.aggregate_residuals)
build_eval_grid = export(ops.build_eval_grid)
select_best = export(ops.select_best)

_cv_residualize = export(ops.cv_residualize)

# Meta steps:

_self = sys.modules[__name__]

fit = partial(_fit, _ops=_self)
fit_eval = partial(ops.fit_eval, _ops=_self)
cv_fit_eval = partial(ops.cv_fit_eval, _ops=_self)
cv_residualize = partial(_cv_residualize, _ops=_self)
