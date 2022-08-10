"""
Tests interoperation of gemz and galp
"""

import numpy as np

import pytest

import galp
import galp.graph

import gemz
from gemz_galp import models

# pylint: disable=redefined-outer-name

@pytest.fixture
def data():
    """
    Random iid gaussian data
    """
    rng = np.random.default_rng(0)

    return rng.normal(0., 1., (2, 11, 13))

@pytest.fixture
async def client():
    """
    Galp client connected to a forked worker
    """
    config = {
        'steps': [ 'gemz_galp.models' ]
        }
    async with galp.local_system(**config) as client:
        yield client

def test_fit_creates_task(data):
    """
    Tests that the fit function creates a galp task instead of returning a
    fitted model.
    """

    train, _ = data

    fitted = models.fit({'model': 'linear'}, train)

    assert isinstance(fitted, galp.graph.Task)

async def test_run_fit(data, client):
    """
    Actually runs a linear fit through gemz
    """
    train, test = data
    model_def = {'model': 'linear'}
    fitted = await client.run(
        models.fit(model_def, train)
        )

    preds = gemz.models.predict_loo(model_def, fitted, test)

    assert preds.shape == test.shape

async def test_run_predict(data, client):
    """
    Fit remotely and predict remotely too by passing model by reference
    """
    train, test = data
    mdef = dict(model='linear')

    fitted = models.fit(mdef, train)
    preds = models.predict_loo(
            mdef, fitted, test
            )

    _preds = await client.run(preds)

    assert _preds.shape == test.shape
