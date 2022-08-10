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
async def model_spec():
    """
    Default test model definition
    """
    return dict(model='linear')

@pytest.fixture
def fitted(data, model_spec):
    """
    Trained linear model (remote)
    """
    train, _ = data
    return models.fit(model_spec, train)

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

async def test_run_fit(data, model_spec, client):
    """
    Actually runs a linear fit through gemz
    """
    train, test = data
    fitted = await client.run(
        models.fit(model_spec, train)
        )

    preds = gemz.models.predict_loo(model_spec, fitted, test)

    assert preds.shape == test.shape

async def test_run_predict(data, fitted, client):
    """
    Fit remotely and predict remotely too by passing model by reference
    """
    _, test = data
    mdef = dict(model='linear')

    preds = models.predict_loo(
            mdef, fitted, test
            )

    _preds = await client.run(preds)

    assert _preds.shape == test.shape

async def test_run_eval(data, model_spec, fitted, client):
    """
    Fit remotely and predict remotely too by passing model by reference
    """
    _, test = data

    rss = await client.run(models.eval_loss(model_spec, fitted, test, 'RSS'))

    assert isinstance(rss, float)

async def test_fit_eval(data, model_spec, client):
    """
    Compound fit-and-eval meta-step
    """
    train, test = data

    fitted, rss = models.fit_eval(model_spec, train, test, 'RSS')

    _rss = await client.run(rss)
    assert isinstance(_rss, float)

    _fitted = await client.run(fitted)

    _rss_2, _fitted_2 = await client.run(rss, fitted)

    assert _rss == _rss_2
