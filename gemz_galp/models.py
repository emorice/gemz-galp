"""
Drop-in for gemz.models with key functions wrapped in galp steps
"""

import gemz.models

from galp.graph import StepSet

export = StepSet()


fit = export.step(gemz.models.fit)
