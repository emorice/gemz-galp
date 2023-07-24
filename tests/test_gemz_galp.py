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
def unsplit_data():
    """
    Random iid gaussian data

    24 (13 + 11) x 17
    """
    rng = np.random.default_rng(0)

    return rng.normal(0., 1., (13 + 11, 17))

@pytest.fixture
def data(unsplit_data):
    """
    Split into 13 x 17 train and 11 x 17 test
    """
    return {
        'train': unsplit_data[:13],
        'test': unsplit_data[13:]
        }

@pytest.fixture
async def model_spec():
    """
    Default test model definition
    """
    return {'model': 'linear'}

@pytest.fixture
def fitted(data, model_spec):
    """
    Trained linear model (remote)
    """
    return models.fit(model_spec, data['train'])

@pytest.fixture
async def client():
    """
    Galp client connected to a forked worker
    """
    config = {
        'steps': [ 'gemz_galp.models' ]
        }
    async with galp.temp_system(**config) as client:
        yield client

@pytest.fixture
async def big_client():
    """
    Galp client connected to many workers
    """
    config = {
        'steps': [ 'gemz_galp.models' ],
        'pool_size': 10,
        }
    async with galp.temp_system(**config) as client:
        yield client


def test_fit_creates_task(data, model_spec):
    """
    Tests that the fit function creates a galp task instead of returning a
    fitted model.
    """

    fitted = models.fit(model_spec, data['train'])

    assert isinstance(fitted, galp.graph.Task)

async def test_run_fit(data, model_spec, client):
    """
    Actually runs a linear fit through gemz
    """
    fitted = await client.run(
        models.fit(model_spec, data['train'])
        )

    preds = gemz.models.predict_loo(model_spec, fitted, data['test'])

    assert preds.shape == data['test'].shape

async def test_run_predict(data, fitted, client):
    """
    Fit remotely and predict remotely too by passing model by reference
    """
    mdef = dict(model='linear')

    preds = models.predict_loo(
            mdef, fitted, data['test']
            )

    _preds = await client.run(preds)

    assert _preds.shape == data['test'].shape

async def test_run_eval(data, model_spec, fitted, client):
    """
    Fit remotely and predict remotely too by passing model by reference
    """
    rss = await client.run(models.eval_loss(model_spec, fitted, data['test'], 'RSS'))

    assert isinstance(rss, float)

async def test_fit_eval(data, model_spec, client):
    """
    Compound fit-and-eval meta-step
    """
    res = models.fit_eval(model_spec, data, 'RSS')

    rss = await client.run(res['loss'])
    assert isinstance(rss, float)

    rss_2 = await client.run(res['loss'])

    assert rss == rss_2

async def test_cv_fit_eval(unsplit_data, model_spec, client):
    """
    Distributed CV-based eval
    """
    # Note: this fetches everything, including data, not what you want outside
    # of tests !
    cvr = await client.run(
        models.cv_fit_eval(model_spec, unsplit_data, 3, 'RSS')
        )

    assert len(cvr['folds']) == 3
    assert all(
        isinstance(f['loss'], float)
        for f in cvr['folds']
        )
    assert isinstance(cvr['loss'], float)

async def test_cv_fit_eval_light(unsplit_data, model_spec, client):
    """
    Distributed CV-based eval, more realistic use
    """
    cvr_task = models.cv_fit_eval(model_spec, unsplit_data, 3, 'RSS')

    # Run everything, but read only the final loss
    total_loss = await client.run(cvr_task['loss'])
    assert isinstance(total_loss, float)

    # Later on, go back to fetch the per-fold losses
    fold_losses = await client.run([
        fold['loss'] for fold in cvr_task['folds']
        ])
    assert len(fold_losses) == 3
    assert all(isinstance(l, float) for l in fold_losses)


def count_done(errtxt):
    """
    Try to parse logs to count the number of task ran

    Clearly not a nice solution, waiting for better
    """
    return sum(
            ('DONE' in line and'gemz.models.ops::fit' in line)
            for line in errtxt.splitlines()
            )

async def test_parallel_cv(unsplit_data, big_client, capsys):
    """
    Calling fit on a cv model generates a graph with one task per point x fold.
    """

    spec = {
        'model': 'cv',
        'inner': { 'model': 'svd' },
        'fold_count': 10,
        'grid': [1, 2, 3, 4, 5]
        }

    task = models.fit(spec, unsplit_data)

    await big_client.run(task)

    errtxt = capsys.readouterr().err

    # 1 cv.fit + 50 submodels + 1 final re-fit
    assert count_done(errtxt) == 52

async def test_parallel_cv_fit_eval(data, big_client, capsys):
    """
    Calling fit_eval on a cv model generates a graph with one task per point x fold.
    """

    spec = {
        'model': 'cv',
        'inner': { 'model': 'svd' },
        'fold_count': 10,
        'grid': [1, 2, 3, 4, 5]
        }

    task = models.fit_eval(spec, data, 'RSS')

    await big_client.run(task)

    errtxt = capsys.readouterr().err

    # 1 cv.fit + 50 submodels + 1 final re-fit
    assert count_done(errtxt) == 52

async def test_parallel_cv_residualize(unsplit_data, big_client, capsys):
    """
    Calling cv_residualize generates a graph with one task per point x fold.
    """

    spec = {'model': 'linear'}

    task = models.cv_residualize(spec, unsplit_data)

    await big_client.run(task)

    errtxt = capsys.readouterr().err

    # 10 submodels
    assert count_done(errtxt) == 10
