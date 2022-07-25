"""
Tests interoperation of gemz and galp
"""

import numpy as np

import pytest

import galp.graph

from gemz_galp import models

@pytest.fixture
def data():
    """
    Random iid gaussian data
    """
    rng = np.random.default_rng(0)

    return rng.normal(0., 1., (2, 11, 13))

def test_fit_creates_task(data):
    """
    Tests that the fit function creates a galp task instead of returning a
    fitted model.
    """

    train, _ = data

    fitted = models.fit({'model': 'linear'}, train.tolist())

    assert isinstance(fitted, galp.graph.Task)
