"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import gemz.models

from galp.graph import StepSet

export = StepSet()

# Wrapping could be automated, but would not be much easier to maintain and
# would break many static analysis programs

fit             = export            (  gemz.models.fit             )
predict_loo     = export            (  gemz.models.predict_loo     )
eval_loss       = export            (  gemz.models.eval_loss       )
fit_eval        = export(items=2)   (  gemz.models.fit_eval        )
