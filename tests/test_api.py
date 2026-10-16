"""Exercise the actual HTTP/job contract with a fake pipeline and real worker."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from contextlib import redirect_stdout
import importlib.util
import io
import logging
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from launcher import main as launcher


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "translator"))
SPEC = importlib.util.spec_from_file_location("translator_contract_test_app", PROJECT / "translator" / "app.py")
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)
api.executor.shutdown(wait=True)
POST_ID = "7659483315203804449"
URL = f"https://www.tiktok.com/@fixture/photo/{POST_ID}"


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        http_log = logging.getLogger("httpx")
        self.addCleanup(http_log.setLevel, http_log.level)
        http_log.setLevel(logging.WARNING)
        with api.jobs_lock:
            api.jobs.clear()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-carousel")
        self.worker_patch = patch.object(api, "executor", self.worker)
        self.worker_patch.start()
        self.settings_patch = patch.object(api, "settings", replace(api.settings, reverse_wait_seconds=2))
        self.settings_patch.start()
        self.client = TestClient(api.app)
        self.release = threading.Event()
        self.pipeline_patches = []

    def tearDown(self):
        self.release.set()
        with api.jobs_lock:
            for job in api.jobs.values():
                job["reverse_event"].set()
        self.worker.shutdown(wait=True, cancel_futures=True)
        for pipeline_patch in reversed(self.pipeline_patches):
            pipeline_patch.stop()
        self.settings_patch.stop()
        self.worker_patch.stop()
        self.client.close()

    def set_pipeline(self, function):
        pipeline_patch = patch.object(api, "run_pipeline", side_effect=function)
        self.pipeline_patches.append(pipeline_patch)
        return pipeline_patch.start()

    def wait_state(self, job_id, expected, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            if response.json()["state"] == expected:
                return response.json()
            time.sleep(0.01)
        self.fail(f"Job did not enter {expected}")

    def start_waiting_job(self):
        self.received = None

        def pipeline(url, force, settings, progress, reverse_callback):
            progress("ocr", "Fixture OCR", 1, 2)
            self.received = reverse_callback(Path("/data/tiktok") / POST_ID, [
                {"position": 1, "clean": "clean/01_CLEAN.jpg"},
                {"position": 2, "clean": "clean/02_CLEAN.jpg"},
            ])
            return {"status": "OK", "post_id": POST_ID, "output_dir": f"/data/tiktok/{POST_ID}"}

        self.set_pipeline(pipeline)
        response = self.client.post("/jobs", json={"url": URL, "reverse_search": True})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        self.wait_state(job_id, "WAITING_REVERSE_SEARCH")
        return job_id

    def test_reverse_event_wait_obeys_remaining_job_budget(self):
        received = []
        def pipeline(url, force, settings, progress, reverse_callback):
            started = time.monotonic()
            answer = reverse_callback(Path('/data/tiktok')/POST_ID,
                        [{'position': 1, 'clean': 'clean/01.jpg',
                          'reverse_query': {'budget_seconds_remaining': .05}}])
            received.append((time.monotonic()-started, answer))
            return {'status': 'OK', 'post_id': POST_ID}
        self.set_pipeline(pipeline)
        response = self.client.post('/jobs', json={'url': URL})
        self.wait_state(response.json()['job_id'], 'COMPLETED')
        elapsed, answers = received[0]
        self.assertLess(elapsed, .75, 'Must use .05s remaining rather than configured 2s wait.')
        self.assertEqual(answers[1]['status'], 'TIMEOUT')

    def test_health_version_and_responsiveness_while_pipeline_is_running(self):
        started = threading.Event()

        def pipeline(*args, **kwargs):
            started.set()
            self.release.wait(10)
            return {"status": "OK", "post_id": POST_ID}

        fake_pipeline = self.set_pipeline(pipeline)
        response = self.client.post("/jobs", json={"url": URL, "reverse_search": False})
        self.assertEqual(response.status_code, 202)
        self.assertTrue(started.wait(1))
        calls = ThreadPoolExecutor(max_workers=1)
        try:
            health = calls.submit(self.client.get, "/health").result(timeout=2)
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["api_version"], 2)
            self.assertEqual(health.json()["status"], "ok")
            self.assertFalse(self.release.is_set())
            self.assertEqual(fake_pipeline.call_count, 1)
        finally:
            self.release.set()
            calls.shutdown(wait=True)
        self.wait_state(response.json()["job_id"], "COMPLETED")

    def test_waiting_request_paths_and_complete_submission_resume_worker(self):
        job_id = self.start_waiting_job()
        job = self.client.get(f"/jobs/{job_id}").json()
        request = job["reverse_search_request"]
        self.assertEqual(Path(request["output_dir"]).name, POST_ID)
        self.assertEqual(request["slides"][0], {"position": 1, "clean": "clean/01_CLEAN.jpg"})
        self.assertNotIn("reverse_event", job)
        self.assertNotIn("reverse_data", job)
        response = self.client.post(f"/jobs/{job_id}/reverse-search", json={"slides": [
            {"position": 1, "status": "NOT_FOUND"},
            {"position": 2, "status": "ERROR", "error": "Lens unavailable"},
        ]})
        self.assertEqual(response.status_code, 200)
        completed = self.wait_state(job_id, "COMPLETED")
        self.assertEqual(completed["result"]["post_id"], POST_ID)
        self.assertEqual(self.received[1]["status"], "NOT_FOUND")
        self.assertEqual(self.received[2]["error"], "Lens unavailable")

    def test_missing_slide_submission_is_rejected_without_resuming_worker(self):
        job_id = self.start_waiting_job()
        response = self.client.post(f"/jobs/{job_id}/reverse-search", json={"slides": [
            {"position": 1, "status": "NOT_FOUND"},
        ]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(f"/jobs/{job_id}").json()["state"], "WAITING_REVERSE_SEARCH")
        self.assertIsNone(self.received)

    def test_duplicate_positions_are_rejected_without_resuming_worker(self):
        job_id = self.start_waiting_job()
        response = self.client.post(f"/jobs/{job_id}/reverse-search", json={"slides": [
            {"position": 1, "status": "NOT_FOUND"},
            {"position": 1, "status": "ERROR"},
        ]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(f"/jobs/{job_id}").json()["state"], "WAITING_REVERSE_SEARCH")

    def test_submission_cannot_be_replayed(self):
        job_id = self.start_waiting_job()
        payload = {"slides": [{"position": 1, "status": "NOT_FOUND"}, {"position": 2, "status": "NOT_FOUND"}]}
        self.assertEqual(self.client.post(f"/jobs/{job_id}/reverse-search", json=payload).status_code, 200)
        self.assertEqual(self.client.post(f"/jobs/{job_id}/reverse-search", json=payload).status_code, 409)

    def test_second_round_waits_for_new_response_and_rejects_stale_round(self):
        received = []
        def pipeline(url, force, settings, progress, reverse_callback):
            for kind in ('clean_full', 'original_full'):
                received.append(reverse_callback(Path('/data/tiktok') / POST_ID, [{
                    'position': 1, 'clean': 'clean/01.jpg', 'original': 'original/01.jpg',
                    'reverse_query': {'kind': kind, 'attempt_index': len(received)+1}}]))
            return {'status': 'OK', 'post_id': POST_ID}
        self.set_pipeline(pipeline)
        job_id = self.client.post('/jobs', json={'url': URL}).json()['job_id']
        first = self.wait_state(job_id, 'WAITING_REVERSE_SEARCH')['reverse_search_request']
        self.assertEqual(first['slides'][0]['original'], 'original/01.jpg')
        payload = {'round_id': first['round_id'], 'slides': [{'position': 1, 'status': 'NOT_FOUND'}]}
        self.assertEqual(self.client.post(f'/jobs/{job_id}/reverse-search', json=payload).status_code, 200)
        deadline = time.monotonic()+2
        while time.monotonic()<deadline:
            job = self.client.get(f'/jobs/{job_id}').json()
            second = job.get('reverse_search_request', {})
            if second.get('round_id') and second['round_id'] != first['round_id']:
                break
            time.sleep(.01)
        else:
            self.fail('Second reverse round did not wait for its own response.')
        self.assertEqual(len(received), 1)
        self.assertEqual(self.client.post(f'/jobs/{job_id}/reverse-search', json=payload).status_code, 409)
        payload['round_id'] = second['round_id']
        self.assertEqual(self.client.post(f'/jobs/{job_id}/reverse-search', json=payload).status_code, 200)
        self.wait_state(job_id, 'COMPLETED')
        self.assertEqual(len(received), 2)

    def test_second_job_is_rejected_until_current_job_finishes(self):
        job_id = self.start_waiting_job()
        response = self.client.post("/jobs", json={"url": URL})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.get(f"/jobs/{job_id}").json()["state"], "WAITING_REVERSE_SEARCH")

    def test_missing_launcher_submission_times_out_to_clean_fallback(self):
        with patch.object(api, "settings", replace(api.settings, reverse_wait_seconds=0)):
            received = {}

            def pipeline(url, force, settings, progress, reverse_callback):
                received.update(reverse_callback(Path("/data/tiktok") / POST_ID, [{"position": 1, "clean": "clean/01_CLEAN.jpg"}]))
                return {"status": "OK", "post_id": POST_ID}

            self.set_pipeline(pipeline)
            response = self.client.post("/jobs", json={"url": URL})
            self.assertEqual(response.status_code, 202)
            self.wait_state(response.json()["job_id"], "COMPLETED")
            self.assertEqual(received[1]["status"], "TIMEOUT")
            self.assertEqual(received[1]["candidates"], [])

    def test_launcher_finishes_when_lens_submits_after_backend_timeout(self):
        created_job = {}
        received = {}

        def pipeline(url, force, settings, progress, reverse_callback):
            received.update(reverse_callback(Path("/data/tiktok") / POST_ID, [{"position": 1, "clean": "clean/01_CLEAN.jpg"}]))
            return {"status": "OK", "post_id": POST_ID, "output_dir": f"/data/tiktok/{POST_ID}"}

        def http(url, method="GET", body=None, **kwargs):
            response = self.client.request(method, url, json=body if method != "GET" else None)
            payload = response.json()
            if response.status_code >= 400:
                raise launcher.LauncherError(str(payload.get("detail")), http_status=response.status_code)
            if method == "POST" and url.endswith("/jobs"):
                created_job["id"] = payload["job_id"]
            return payload

        def delayed_lens(*args, **kwargs):
            self.wait_state(created_job["id"], "COMPLETED")
            return [{"position": 1, "status": "NOT_FOUND"}]

        self.set_pipeline(pipeline)
        config = launcher.Config.from_project(PROJECT)
        config.api_url = "http://testserver"
        config.poll_interval = 0.001
        with patch.object(api, "settings", replace(api.settings, reverse_wait_seconds=1)), patch.object(
            launcher, "http_json", side_effect=http
        ), patch.object(launcher, "run_reverse_search", side_effect=delayed_lens) as browser, redirect_stdout(io.StringIO()):
            result = launcher.process_job(config, URL)
        self.assertEqual(result["post_id"], POST_ID)
        browser.assert_called_once()
        self.assertEqual(received[1]["status"], "TIMEOUT")

    def test_pipeline_failure_has_error_state_and_message_in_result(self):
        self.set_pipeline(lambda *args, **kwargs: {"status": "ERROR", "error": "Ollama fixture unavailable"})
        response = self.client.post("/jobs", json={"url": URL, "reverse_search": False})
        job = self.wait_state(response.json()["job_id"], "ERROR")
        self.assertEqual(job["result"]["error"], "Ollama fixture unavailable")

    def test_invalid_domain_and_unknown_job_have_clear_http_errors(self):
        response = self.client.post("/jobs", json={"url": "https://tiktok.com.attacker.invalid/post"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/jobs/nonexistent").status_code, 404)


if __name__ == "__main__":
    unittest.main()
