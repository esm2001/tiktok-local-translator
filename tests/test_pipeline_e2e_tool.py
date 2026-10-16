"""Isolation and honest evidence boundaries for the real E2E command."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_pipeline_e2e import (FixtureReplay, cached_repeat, clone_existing_outputs, count_pipeline_work,
                                   existing_snapshot, known_urls, load_inputs, main, no_network_downloads,
                                   raw_ocr_evidence, safe_sandbox, score_manifest)


def test_pipeline_sandbox_cannot_overwrite_existing_user_post(tmp_path):
    for output in (tmp_path, tmp_path / '1234567890123456789', tmp_path / '.cache/regression/e2e', tmp_path.parent / 'external'):
        with pytest.raises(ValueError, match='new run'):
            safe_sandbox(tmp_path, output)
    target = tmp_path / '.cache/regression/e2e/new-run/posts'
    assert safe_sandbox(tmp_path, target) == target.resolve()
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        safe_sandbox(tmp_path, target)


def test_existing_snapshot_detects_change_to_source_and_text(tmp_path):
    post = tmp_path / '1234567890123456789'
    (post / 'original').mkdir(parents=True)
    (post / 'original/01.jpg').write_bytes(b'original')
    (post / 'traducciones.txt').write_text('before', encoding='utf-8')
    before = existing_snapshot(tmp_path, [post.name])
    (post / 'traducciones.txt').write_text('after', encoding='utf-8')
    after = existing_snapshot(tmp_path, [post.name])
    assert before != after
    assert before['1234567890123456789/original/01.jpg'] == after['1234567890123456789/original/01.jpg']


def test_fixture_transfer_rejects_changed_evidence(tmp_path):
    original = tmp_path / 'source.jpg'
    original.write_bytes(b'fixture-evidence')
    replay = FixtureReplay({'slides': {}}, tmp_path)
    candidate = replay.candidate('1234567890123456789', 1, 'source', original, {})
    target = tmp_path / 'copy.jpg'
    actual = replay.download(candidate, target)
    assert actual['bytes'] == len(b'fixture-evidence')
    assert target.read_bytes() == original.read_bytes()
    original.write_bytes(b'changed-after-registration')
    rejected = replay.download(candidate, target)
    assert 'changed' in rejected['error']
    assert not target.exists()


def test_replay_does_not_accept_unregistered_or_external_candidate(tmp_path):
    replay = FixtureReplay({'slides': {}}, tmp_path)
    rejected = replay.download({'image_url': 'https://example.com/arbitrary.jpg'}, tmp_path / 'download.jpg')
    assert 'Unregistered' in rejected['error']
    with pytest.raises(ValueError, match='outside media root'):
        replay.candidate('post', 1, 'source', tmp_path.parent / 'external.jpg', {})


def test_known_url_requires_matching_verified_post_identifier(tmp_path):
    post = tmp_path / '1234567890123456789' / 'metadata'
    post.mkdir(parents=True)
    (post / 'manifest.json').write_text(json.dumps({'resolved_url': 'https://www.tiktok.com/@name/photo/9999999999999999999'}))
    dataset = {'slides': {'one': {'post_id': '1234567890123456789'}}}
    with pytest.raises(ValueError, match='verified canonical'):
        known_urls(dataset, tmp_path)


def test_new_fixture_can_record_its_verified_url_without_rewriting_user_outputs(tmp_path):
    url = 'https://www.tiktok.com/@name/photo/1234567890123456789'
    dataset = {'slides': {'one': {'post_id': '1234567890123456789', 'canonical_url': url}}}
    assert known_urls(dataset, tmp_path) == [url]
    assert not (tmp_path / '1234567890123456789').exists()


def test_unlabelled_new_carousel_does_not_receive_invented_ground_truth(tmp_path):
    report = score_manifest({'post_id': 'new-post', 'slides': [{'position': 1}]}, {'slides': {}}, tmp_path)
    assert report['expected_slides'] == 0
    assert report['cases'] == [{'position': 1, 'ground_truth': 'NOT_AVAILABLE'}]
    assert report['semantic_human_review'] == 'PENDING'


def test_natural_logo_text_is_not_an_invented_editorial_translation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('src.image.cleaning.read_image', lambda path: SimpleNamespace(shape=(100, 100, 3)))
    case = {'post_id': 'post', 'position': 1, 'image_sha256': 'same', 'expected_text': '',
            'expected_regions': [], 'expected_editorial_empty': True, 'tags': ['fresh_carousel']}
    lines = [{'text': 'HULL CITY', 'polygon': []}]
    slide = {'position': 1, 'original': 'original/01.jpg', 'original_sha256': 'same',
             'ocr': {'lines': lines, 'editorial_lines': lines, 'polygons': [], 'reconstructed': 'HULL CITY'},
             'translation': {'spanish': 'HULL CITY'}}
    report = score_manifest({'post_id': 'post', 'slides': [slide]}, {'slides': {'holdout': case}}, tmp_path)
    row = report['cases'][0]
    assert row['observed_raw_lines'] == row['observed_editorial_lines'] == 1
    assert row['cohort'] == 'HOLDOUT'
    assert row['reconstruction'] == row['translation_clause_proxy'] == {'status': 'NO_EDITORIAL_OVERLAY_NOT_SCORED'}


def test_holdout_cannot_overwrite_regression_truth(tmp_path):
    dataset = {'media_root': '../media', 'slides': {'same-case': {'expected_text': 'reviewed'}}}
    expected, holdout = tmp_path / 'expected.json', tmp_path / 'holdout.json'
    expected.write_text(json.dumps(dataset), encoding='utf-8')
    holdout.write_text(json.dumps({**dataset, 'slides': {'same-case': {'expected_text': 'replacement'}}}), encoding='utf-8')
    with pytest.raises(ValueError, match='overlaps regression'):
        load_inputs(expected, holdout)
    assert json.loads(expected.read_text(encoding='utf-8')) == dataset


def prior_output(root):
    source = root / '.cache/regression/e2e/prior/posts'
    post = source / '1234567890123456789'
    (post / 'original').mkdir(parents=True)
    (post / 'metadata').mkdir()
    (post / 'original/01.jpg').write_bytes(b'original-photograph')
    (post / 'traducciones.txt').write_text('known failed translation', encoding='utf-8')
    manifest = {'post_id': post.name, 'resolved_url': 'https://www.tiktok.com/@name/photo/' + post.name,
                'slides': [{'position': 1, 'best_image': 'clean/01_CLEAN.jpg',
                            'ocr': {'lines': [{'text': 'literal evidence'}], 'polygons': [], 'primary_lines': []}}]}
    (post / 'metadata/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return source, post, manifest


def test_revalidation_clone_preserves_failed_evidence_and_original_hashes(tmp_path):
    source, post, _ = prior_output(tmp_path)
    before = existing_snapshot(source, [post.name])
    target = tmp_path / '.cache/regression/e2e/fresh/posts'
    result = clone_existing_outputs(tmp_path, source.relative_to(tmp_path), target, [post.name])
    assert result['prior_hashes'] == before == existing_snapshot(target, [post.name])
    assert result['clone_matches_prior_evidence'] is True
    (target / post.name / 'traducciones.txt').write_text('new candidate', encoding='utf-8')
    assert existing_snapshot(source, [post.name]) == before
    assert existing_snapshot(target, [post.name]) != before
    assert (target / post.name / 'original/01.jpg').read_bytes() == (post / 'original/01.jpg').read_bytes()


def test_revalidation_clone_rejects_external_existing_and_nested_targets(tmp_path):
    source, post, _ = prior_output(tmp_path)
    with pytest.raises(ValueError, match='independent'):
        clone_existing_outputs(tmp_path, source, source / 'nested/posts', [post.name])
    with pytest.raises(ValueError, match='already exists'):
        clone_existing_outputs(tmp_path, source, source, [post.name])
    with pytest.raises(ValueError, match='must exist inside'):
        clone_existing_outputs(tmp_path, tmp_path, tmp_path / '.cache/regression/e2e/fresh/posts', [post.name])
    assert not (tmp_path / '.cache/regression/e2e/fresh/posts').exists()


def test_revalidation_missing_manifest_or_originals_fails_before_copy(tmp_path):
    source, post, _ = prior_output(tmp_path)
    target = tmp_path / '.cache/regression/e2e/fresh/posts'
    (post / 'original/01.jpg').unlink()
    with pytest.raises(ValueError, match='empty'):
        clone_existing_outputs(tmp_path, source, target, [post.name])
    assert not target.exists()
    with pytest.raises(ValueError, match='manifest missing'):
        clone_existing_outputs(tmp_path, source, target, ['9876543210987654321'])


def test_revalidation_clone_detects_corrupted_copy_without_modifying_prior(tmp_path, monkeypatch):
    import validate_pipeline_e2e as tool
    source, post, _ = prior_output(tmp_path)
    target = tmp_path / '.cache/regression/e2e/fresh/posts'
    before = existing_snapshot(source, [post.name])
    copytree = tool.shutil.copytree

    def corrupted_copy(origin, destination, *args, **kwargs):
        result = copytree(origin, destination, *args, **kwargs)
        if Path(destination).name == post.name:
            (Path(destination) / 'original/01.jpg').write_bytes(b'corrupted-copy')
        return result

    monkeypatch.setattr(tool.shutil, 'copytree', corrupted_copy)
    with pytest.raises(ValueError, match='hash verification failed'):
        clone_existing_outputs(tmp_path, source, target, [post.name])
    assert existing_snapshot(source, [post.name]) == before


def test_report_cannot_overwrite_previous_evidence_even_under_cache(tmp_path):
    source, post, _ = prior_output(tmp_path)
    dataset = tmp_path / 'expected.json'
    dataset.write_text(json.dumps({'media_root': '.', 'slides': {}}), encoding='utf-8')
    manifest = post / 'metadata/manifest.json'
    before = manifest.read_bytes()
    with pytest.raises(SystemExit) as rejected:
        main(['--dataset', str(dataset), '--media-root', str(tmp_path), '--revalidate-existing',
              source.relative_to(tmp_path).as_posix(), '--output', str(manifest)])
    assert rejected.value.code == 2
    assert manifest.read_bytes() == before


def fake_pipeline():
    class Translator:
        def translate(self, value):
            return self._chat(value)

        def _chat(self, value):
            return 'delegated:' + value

    return SimpleNamespace(download_images=lambda *a, **k: 'downloaded',
                           resolve_tiktok_url=lambda value: value, ocr_image=lambda *a, **k: 'recognized',
                           OllamaTranslator=Translator)


def test_work_counters_delegate_methods_and_nested_chat(monkeypatch):
    import src.recovery as recovery
    pipeline = fake_pipeline()
    monkeypatch.setattr(recovery, 'ocr_image', lambda *a, **k: 'source-recognized')
    with count_pipeline_work(pipeline) as counts:
        assert pipeline.download_images('url') == 'downloaded'
        assert pipeline.ocr_image('photo') == 'recognized'
        assert recovery.ocr_image('source') == 'source-recognized'
        assert pipeline.OllamaTranslator().translate('text') == 'delegated:text'
    assert counts == {'download': 1, 'ocr': 1, 'source_ocr': 1, 'translation': 1, 'ollama_chat': 1}
    assert pipeline.OllamaTranslator().translate('text') == 'delegated:text'
    assert counts['translation'] == 1  # instrumentation was removed


def test_revalidation_guard_cannot_silently_download_new_evidence():
    import src.recovery as recovery
    pipeline = fake_pipeline()
    with no_network_downloads(pipeline) as attempts:
        for function, arguments in [(pipeline.download_images, ('url',)),
                                    (pipeline.resolve_tiktok_url, ('url',)),
                                    (recovery.download_candidate, ({}, Path('new.jpg')))]:
            with pytest.raises(RuntimeError, match='Unexpected network operation'):
                function(*arguments)
    assert attempts == ['gallery_download', 'url_resolution', 'candidate_download']
    assert pipeline.download_images('url') == 'downloaded'


@pytest.mark.parametrize('unexpected_call', [False, True])
def test_real_cached_probe_separates_timing_and_detects_work_despite_cache_claim(tmp_path, unexpected_call):
    source, post, manifest = prior_output(tmp_path)
    pipeline = fake_pipeline()
    observed = []

    def run(url, **kwargs):
        observed.append(kwargs)
        if unexpected_call:
            pipeline.OllamaTranslator().translate('unexpected')
        return {'status': 'OK_CACHED', 'cached': True, 'timings': {'total': 0.125}}

    pipeline.run_pipeline = run
    replay = FixtureReplay({'slides': {}}, tmp_path)
    result = cached_repeat(pipeline, manifest['resolved_url'], SimpleNamespace(output_base=source),
                           lambda *a: None, replay, post, manifest, reverse_callback=None)
    assert observed[0]['force'] is False and observed[0]['reverse_callback'] is None
    assert result['pipeline_reported_seconds'] == 0.125
    assert result['seconds'] >= result['pipeline_wall_seconds'] >= 0
    assert result['managed_artifacts_unchanged'] is True
    assert result['source_pointers_unchanged'] is True
    assert result['additional_stage_calls']['translation'] == int(unexpected_call)
    assert result['additional_stage_calls']['ollama_chat'] == int(unexpected_call)
    assert result['passed'] is (not unexpected_call)


def test_raw_evidence_comparison_allows_reclassified_background_but_catches_text_edit():
    old = {'slides': [{'position': 1, 'ocr': {'lines': [{'text': 'M'}], 'polygons': [[[1, 2]]],
                                           'primary_lines': [{'text': 'M'}], 'editorial_lines': [{'text': 'M'}]}}]}
    new = json.loads(json.dumps(old))
    new['slides'][0]['ocr']['editorial_lines'] = []
    assert raw_ocr_evidence(old) == raw_ocr_evidence(new)
    new['slides'][0]['ocr']['lines'][0]['text'] = 'invented'
    assert raw_ocr_evidence(old) != raw_ocr_evidence(new)


def test_revalidation_does_not_claim_prior_negative_attempt_as_a_fresh_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr('src.image.cleaning.read_image', lambda path: SimpleNamespace(shape=(100, 100, 3)))
    case = {'post_id': 'post', 'position': 1, 'image_sha256': 'same', 'expected_text': '',
            'expected_regions': [], 'expected_editorial_empty': True}
    slide = {'position': 1, 'original': 'original/01.jpg', 'original_sha256': 'same',
             'ocr': {'lines': [], 'polygons': []},
             'reverse_search': {'status': 'MATCH', 'cache_revalidated': True, 'queries_attempted': 1,
                                'attempts': [{'query': {'kind': 'clean_full'}, 'status': 'DISABLED', 'candidates': []}]}}
    prior = {'slides': [{'position': 1, 'reverse_search': {'queries_attempted': 2,
                        'attempts': [{'query': {'kind': 'clean_full'}, 'candidates': [{'accepted': False}]}]}}]}
    row = score_manifest({'post_id': 'post', 'slides': [slide]}, {'slides': {'source': case}}, tmp_path,
                         revalidation=True, prior_manifest=prior)['cases'][0]
    assert row['source_cache_revalidated'] is True
    assert row['negative_candidate_rejected'] is None and row['queries_attempted'] == 0
    assert row['prior_negative_candidate_rejected'] is True and row['prior_queries_attempted'] == 2
    assert row['backend_local_attempt_records'] == 1
    assert row['discovery_evidence_scope'] == 'PRIOR_RUN_ONLY_NO_DISCOVERY_THIS_RUN'
