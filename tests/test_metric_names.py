"""Producer coverage, duplicate proofs, logging boundaries and offline panels."""
from dataclasses import replace
import json
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from orbitol import contracts as C, eval_protocol, metric_names as names, posthoc_eval, posthoc_queue, symmetry_panel, train
from orbitol.eval_adapter import _sample_items
from orbitol.eval_diagnostics import variant_metrics
from orbitol.eval_protocol import relax_all, score
from orbitol.posthoc_io import read_json, write_json
from test_eval_protocol import Relaxer, evaluator, protocol  # noqa: F401
from test_posthoc_queue import setup as queue_setup  # noqa: F401
from test_train import cfg, fixture_data, trainer  # noqa: F401
from tools.wandb_workspace import build_sections, build_workspace

ROOT = Path(__file__).resolve().parents[1]
HISTORIES = [Path(C.RUNS_ROOT) / run / 'metrics.jsonl'
             for run in ('orbitol_alex_x_tw025_d256_layergem_fast_2m', 'orbitol_alex_x_tw025_d256_2m')]


@pytest.fixture(autouse=True)
def direct_fixture_symmetry(monkeypatch):
    # These tiny known-safe cells need no worker startup; keep real spglib results.
    def detect(structures, symprecs=symmetry_panel.SYMPRECS, **kwargs):
        return np.array([[symmetry_panel.detect_group(s, p) for p in symprecs]
                         for s in structures], dtype=np.int16).reshape(-1, len(symprecs))
    monkeypatch.setattr(symmetry_panel, 'detect_many', detect)
    monkeypatch.setattr(eval_protocol, 'detect_many', detect)


def checked_view(row, *, legacy=False):
    targets = [names._name(key, row, legacy) for key in row]
    targets = [key for key in targets if key is not None]
    assert len(targets) == len(set(targets)), targets
    result = names.log_view(row, legacy_subset=legacy)
    assert len(result) == len(targets)
    assert not any(key.startswith('other/') for key in result)
    assert all(key.count('/') == 1 or key in {'step', 'train_step'} for key in result)
    assert names.canonical(result) == result
    return result


def assert_duplicates(row):
    """Independent numerical proofs for every dropped key in an all-pool row."""
    twins = {
        'joint/SUN': 'sample_ema/SUN', 'joint/MSUN': 'sample_ema/MSUN',
        'joint/checked_count': 'sample_ema/stability/attempt_count',
        'joint/relaxed_count': 'sample_ema/stability/relaxed_ok_count',
        'diag/e_above_hull_median': 'sample_ema/stability/e_above_hull_median',
        'sample_ema/all_relaxed/thermo_checked_total': 'sample_ema/stability/constructed_count',
        'sample_ema/all_relaxed/thermo_success_total': 'sample_ema/stability/hull_ok_count',
    }
    stable = {k.removeprefix('sample_ema/stability/'): v for k, v in row.items()
              if k.startswith('sample_ema/stability/')}
    expected = {key: row[twin] for key, twin in twins.items() if key in row}
    prefix = 'sample_ema/all_relaxed/thermo_'
    expected[prefix + 'failed_total'] = stable['relax_failure_count'] + stable['hull_failure_count']
    if stable['constructed_count']:
        for kind, value in (
            ('success', stable['hull_ok_count']),
            ('failure', expected[prefix + 'failed_total']),
            ('stable', stable['stable_rate'] * stable['attempt_count']),
            ('metastable', stable['metastable_rate'] * stable['attempt_count']),
        ):
            expected[prefix + kind + '_rate'] = value / stable['constructed_count']
    for suffix, twin in names.THERMO_COPIES.items():
        key = 'sample_ema/all_relaxed/nequip_stability/' + suffix
        if key in row:
            expected[key] = row[prefix + twin]
    for label, count in (('raw', stable['attempt_count']), ('relaxed', stable['relaxed_ok_count'])):
        for tol in names.SYMMETRY_TOLERANCES:
            pre = f'sym/{label}/{tol}/'
            expected[pre + 'count'] = count
            failed = row[pre + 'failure_rate'] * count if count else 0
            expected[pre + 'failure_count'] = failed
            expected[pre + 'detected_count'] = count - failed
            if pre + 'retention' in row:
                expected[pre + 'retention_total'] = count
                expected[pre + 'retention_count'] = row[pre + 'retention'] * count if count else 0
    dropped = {key for key in row if names._name(key, row, False) is None}
    assert dropped <= expected.keys(), dropped - expected.keys()
    for key in dropped:
        if row[key] is None or (isinstance(row[key], float) and math.isnan(row[key])):
            assert expected[key] is None or math.isnan(expected[key]), key
        else:
            assert row[key] == pytest.approx(expected[key], abs=1e-10), key


def test_score_symmetry_variants_and_duplicate_proofs(protocol):
    runner = evaluator(protocol)
    items = _sample_items(protocol.rows)
    records = relax_all(items, Relaxer(), 500)
    metrics, rows = score(items, records, references=runner.references, ppd=runner.ppd,
                          settings=runner.settings, tag=runner.tag, step=0, requested_G=torch.ones(8))
    logged = checked_view(metrics)
    assert logged['eval/MSUN'] == 2 / 8
    assert logged['eval_count/attempts'] == 8
    assert logged['eval_hull/e_above_hull_mean'] == pytest.approx(.056)
    assert logged['eval_count/relax_divergence_total'] == 1
    assert_duplicates(metrics)
    assert any(k.startswith('sym_raw/') for k in logged)
    assert any(k.startswith('sym_relaxed/') for k in logged)
    # Standalone diagnostic producers keep their median if its twin is absent.
    diagnostics = variant_metrics([dict(
        is_un=row['relaxed_is_un'], raw_valid=row['raw_valid'], stable=row['stable'],
        metastable=row['metastable'], e_above_hull=row['e_above_hull'],
        relaxation_outcome='success' if row['relax_status'] == 'ok' and row['hull_failure_reason'] is None
        else 'failed',
    ) for row in rows], checked_count=len(rows))
    assert checked_view(diagnostics)['eval_hull/e_above_hull_median'] == logged['eval_hull/e_above_hull_median']
    for key in diagnostics:
        assert diagnostics[key] == pytest.approx(metrics[key])


def test_real_legacy_producer_keeps_joint_metrics(protocol):
    runner = evaluator(protocol, sun_k=2)
    metrics = runner.evaluate(protocol.rows, step=0, requested_G=torch.ones(8))
    metrics.update(variant_metrics(runner.relaxation_diagnostics(), checked_count=metrics['joint/checked_count']))
    logged = checked_view(metrics, legacy=True)
    for key, target in names.LEGACY_JOINT.items():
        assert logged[target] == metrics[key]
    assert any(k.startswith('eval_legacy/sun_') for k in logged)
    assert names.canonical(metrics) == logged


def test_divergence_is_not_only_force_divergence():
    from src.utils.stability_logger import StabilityLogger
    assert StabilityLogger._thermo_exception_reason(RuntimeError('force divergence')) == ('relax_divergence', True)
    assert StabilityLogger._thermo_exception_reason(RuntimeError('non-finite energy')) == ('relax_non_finite', True)
    row = {'sample_ema/all_relaxed/thermo_divergence_total': 2,
           'sample_ema/all_relaxed/thermo_fail/relax_divergence': 1,
           'sample_ema/all_relaxed/thermo_fail/relax_non_finite': 1}
    result = checked_view(row)
    assert result['eval_count/relax_divergence_total'] == 2
    assert result['eval_count/relax_fail_relax_divergence'] == 1


def test_training_validation_perf_and_logging(cfg, fixture_data, monkeypatch):
    run = trainer(replace(cfg, max_steps=1, val_every=1, ckpt_every=0, sun_k=0), fixture_data, 'names')
    inputs, remote, callbacks = [], [], []
    original = run.log

    def capture(row):
        inputs.append(dict(row))
        checked_view(row)
        original(row)
        assert row == inputs[-1]  # Mapping cannot mutate producers or callbacks.

    run.log = capture
    run.wandb_run = SimpleNamespace(log=lambda row, step: remote.append((step, dict(row))), finish=lambda: None)
    try:
        run.run(on_step=lambda step, losses: callbacks.append(set(losses)))
        assert callbacks == [{'total', 'A', 'F', 'Y'}]
        merged = {k: v for row in inputs for k, v in row.items()}
        assert {'loss/total', 'lr', 'stats/type_acc', 'sigma/mean', 'perf/steps_per_second',
                'val/loss_total', 'val/sigma_0.1/coord_ratio/d1'} <= merged.keys()
        local = [json.loads(line) for line in (run.run_dir / 'metrics.jsonl').read_text().splitlines()]
        assert local == [{'step': step, **row} for step, row in remote]
        assert 'train/loss_total' in local[0] and 'loss/total' not in local[0]
        assert 'val_sigma/loss_coord_s0.1' in local[0]

        result = dict(rows={}, metadata={}, requested_G=torch.ones(2),
                      contains_a=torch.tensor([True, False]), failure_counts={'invalid_species': 1})
        monkeypatch.setattr(train, 'sample_pool', lambda *a, **kw: result)
        run.evaluator = SimpleNamespace(tag='sample_ema', evaluate=lambda *a, **kw: {'sample_ema/MSUN': .25})
        raw = run.evaluate_samples()
        mapped = checked_view(raw)
        assert mapped['eval/best_MSUN'] == .25
        assert mapped['decode/contains_a_rate'] == .5
        assert mapped['decode/fail_invalid_species'] == 1
        assert mapped['decode/fail_non_finite'] == 0
        assert set(names.DECODE_REASONS) == set(train.DECODE_FAILURES)
        checked_view(posthoc_eval.sample_metrics(result))

        from orbitol import eval_jobs, eval_snapshot
        run.cfg = replace(run.cfg, eval_mode='async')
        monkeypatch.setattr(eval_snapshot, 'write_eval_snapshot', lambda *a: None)
        monkeypatch.setattr(eval_snapshot, 'pending_eval_snapshots', lambda *a: [])
        for job in ('123', ''):
            monkeypatch.setattr(eval_jobs, 'submit_eval', lambda *a, **kw: job)
            status = checked_view(run.evaluate_samples())
            assert status['status/eval_snapshot_step'] == run.step
            if job:
                assert status['status/eval_job_id'] == job
            else:
                assert status['status/eval_submit_failed'] == 1
        monkeypatch.setattr(eval_snapshot, 'pending_eval_snapshots', lambda *a: [1, 2])
        assert checked_view(run.evaluate_samples()) == {'status/eval_skipped_backlog': 1}
    finally:
        run.close()


@pytest.mark.parametrize('legacy', [False, True])
def test_posthoc_logging_boundaries(queue_setup, monkeypatch, legacy):
    s = queue_setup
    s.snapshot(s.runs[0], 20000)
    directory = s.pool(s.runs[0], 20000)
    row = dict(zip(('sample_ema/MSUN', 'joint/MSUN', 'posthoc/sample_seconds',
                    'posthoc/evaluate_seconds', 'orbit/eval_snapshot_step', 'orbit/eval_job_id',
                    'orbit/eval_submit_failed', 'orbit/eval_skipped_backlog', 'orbit/nequip_error'),
                   (.25, .5 if legacy else .25, 2., 3., 20000, '12', 0, 0, 0)))
    settings = dict(thermo={}, sun_k=2 if legacy else 0)
    write_json(directory / 'evaluation_settings.json', settings)
    write_json(directory / 'metrics.json', row)
    logged = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=lambda **kw: SimpleNamespace(step=0, finish=lambda: None),
        define_metric=lambda *a, **kw: None, log=lambda values, **kw: logged.append(values)))
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    posthoc_queue.log_wandb(s.args, s.runs[0], directory, row)
    pool = read_json(directory / 'pool.json')
    assert logged[-1] == checked_view({**row, 'train_step': 20000,
                                      'posthoc/pool_sha256': pool['samples_sha256']}, legacy=legacy)
    args = posthoc_eval.parse_args(['--kind', 'crystalite', '--stage', 'evaluate', '--out', str(directory), '--wandb-name', 'test'])
    posthoc_eval._log_wandb(args, row, 20000, settings, pool)
    assert logged[-1] == checked_view({**row, 'train_step': 20000}, legacy=legacy)
    assert ('eval_legacy/joint_MSUN' in logged[-1]) == legacy
    assert read_json(directory / 'metrics.json') == row


@pytest.mark.parametrize('path', HISTORIES, ids=lambda p: p.parent.name)
def test_real_history_inventory_and_duplicates(path):
    if not path.is_file():
        pytest.skip(f'History unavailable: {path}')
    eval_count = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            logged = checked_view(row)
            assert names.canonical(row) == logged
            if 'sample_ema/MSUN' in row:
                eval_count += 1
                assert logged['eval/MSUN'] == row['sample_ema/MSUN']
                assert logged['eval/MSUN_raw_UN'] == row.get('sample_ema/MSUN_raw_UN', row.get('sample_ema/MSUN_raw'))
                assert logged['eval_count/attempts'] == row['joint/checked_count']
                assert_duplicates(row)
    assert eval_count > 0


def test_future_families_and_unknown_keys(caplog, monkeypatch):
    row = {key: i for i, key in enumerate(names.expected_internal_keys())
           if key not in {'sample_ema/SUN_raw', 'sample_ema/MSUN_raw'}}
    checked_view(row)
    for metric in ('SUN', 'MSUN'):
        for spelling in ('raw', 'raw_UN'):
            assert names.log_view({f'sample_ema/{metric}_{spelling}': .3}) == {f'eval/{metric}_raw_UN': .3}
    monkeypatch.setattr(names, '_warned_unknown', set())
    with caplog.at_level(logging.WARNING, logger=names.__name__):
        result = names.log_view({'future/deep/key': 0, 'mystery': 'text'})
        assert result == {'other/future__deep__key': 0, 'other/mystery': 'text'}
        assert names.log_view({'future/deep/key': 1}) == {'other/future__deep__key': 1}
        assert names.canonical(result) == result
    assert len(caplog.records) == 2


def test_collision_keeps_first_and_warns(caplog, monkeypatch):
    monkeypatch.setattr(names, '_warned_collisions', set())
    with caplog.at_level(logging.WARNING, logger=names.__name__):
        result = names.log_view({'sample_ema/MSUN_raw': .1, 'sample_ema/MSUN_raw_UN': .2})
        assert result == {'eval/MSUN_raw_UN': .1}
    assert 'collision' in caplog.text and 'keeping first' in caplog.text


def test_offline_workspace_and_cli(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Workspace construction must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    sections = build_sections()
    assert [s['name'] for s in sections] == [
        'Headline', 'Relaxation', 'Relaxation detail', 'Hull', 'Training', 'Validation',
        'Perf', 'Counts', 'Symmetry', 'Decode', 'Status']
    assert [s['is_open'] for s in sections] == [True, True, False, True, True, True] + [False] * 5
    headline = sections[0]['panels']
    assert headline[0]['y'] == ['eval/MSUN', 'sample_ema/MSUN']
    assert headline[2]['y'][:2] == ['eval/MSUN_raw_UN', 'sample_ema/MSUN_raw']
    assert len(headline) == 10
    assert sections[1]['panels'][0]['y'] == ['eval_relax/rms_disp_median']
    assert sections[2]['panels'][0]['y'] == [f'eval_relax/rms_disp_{s}' for s in ('median', 'mean', 'p90')]
    relax_keys = {name for key in names.relax_internal_keys() for name in names.log_view({key: 0})}
    panel_keys = {y for p in sections[1]['panels'] + sections[2]['panels'] for y in p['y']}
    assert relax_keys == {y for y in panel_keys if y.startswith('eval_relax/')}
    assert all(len(p['y']) == 7 for p in sections[5]['panels'] if p['y'][0].startswith(('val_sigma/', 'val_coord_class/')))
    assert all(len(p['y']) == (7 if 'crystal systems' in p['title'] else 2) for p in sections[8]['panels'])
    assert sections[9]['panels'][0]['y'] == [f'decode/fail_{reason}' for reason in names.DECODE_REASONS]
    workspace = build_workspace(entity='test-entity', project='orbitol_alex_mp20')
    assert [s.name for s in workspace.sections] == [s['name'] for s in sections]
    # Exercise the installed SDK serializer, without its save/network operation.
    workspace._to_model()
    output = subprocess.check_output([sys.executable, str(ROOT / 'tools/wandb_workspace.py'),
                                      '--project', 'orbitol_alex_mp20', '--dry-run'], text=True,
                                     env=dict(os.environ, ORBITOL_WANDB_ENTITY='test-entity'))
    assert json.loads(output)['entity'] == 'test-entity'
    assert json.loads(output)['sections'] == sections


def test_relax_catalogue_matches_producer():
    """The view's eval_relax panels name exactly what eval_protocol._relax_metrics emits."""
    rows = []
    for i in range(8):
        rows.append({"relax_status": "ok", "initial_energy_eV": -1.0 - i, "MSUN": i % 2 == 0,
                     "raw_space_group_0.1": 225, "relaxed_space_group_0.1": 225 if i < 4 else 221,
                     "sg_kept": i < 4, "final_fmax": 0.01 * i, "converged": i < 5,
                     "e_above_hull_oneshot": 0.05 * i, "SUN_oneshot": False, "MSUN_oneshot": i == 0,
                     **{q: 0.1 * (i + 1) for q in names.RELAX_QUANTITIES if q != "final_fmax"}})
    produced = {key for key in eval_protocol._relax_metrics(rows, "sample_ema") if key.startswith("sample_ema/relax/")}
    assert produced == set(names.relax_internal_keys())
    assert not any(name.startswith("other/") for name in names.log_view(dict.fromkeys(produced, 0.0)))
