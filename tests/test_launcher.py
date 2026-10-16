"""Host launcher checks that do not start Docker, Ollama or a real browser."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from launcher import main as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "TikTokTranslator"
        translator = self.project / "translator"
        translator.mkdir(parents=True)
        (translator / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
        (translator / "requirements.txt").write_text("fastapi==0.120.0\n", encoding="utf-8")
        (translator / "app.py").write_text("API_VERSION = 2\n", encoding="utf-8")
        (self.project / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        (self.project / "requirements-launcher.txt").write_text("playwright>=1.51,<2\n", encoding="utf-8")
        self.config = launcher.Config.from_project(self.project, {})
        self.config.poll_interval = 0.001

    def seed_backend_state(self):
        launcher.save_backend_state(
            self.config,
            launcher.dependency_fingerprint(self.project),
            launcher.source_fingerprint(self.project),
        )

    def fake_docker(self, *, image_exists=True):
        commands = []
        state = {"image": image_exists}

        def run(args, **kwargs):
            commands.append(args)
            if args[1:3] == ["image", "inspect"]:
                code = 0 if state["image"] else 1
            else:
                code = 0
                if "build" in args:
                    state["image"] = True
            return subprocess.CompletedProcess(args, code, "", "")

        return commands, run

    def test_default_host_output_uses_real_sibling_folder(self):
        self.assertEqual(self.config.output_dir, self.project.parent / "TikTokTraducidos")
        self.assertEqual(self.config.command_env["OUTPUT_HOST_DIR"], str(self.config.output_dir))

    def test_first_launch_configuration_uses_example_without_manual_setup(self):
        example = self.project / ".env.example"
        content = "OLLAMA_MODEL=qwen2.5:3b\nOUTPUT_HOST_DIR=../Resultados\n"
        example.write_text(content, encoding="utf-8")
        self.assertTrue(launcher.initialize_env(self.project))
        config = launcher.Config.from_project(self.project, {})
        self.assertEqual(config.model, "qwen2.5:3b")
        self.assertEqual(config.output_dir, self.project.parent / "Resultados")
        (self.project / ".env").write_text("OLLAMA_MODEL=custom-model\n", encoding="utf-8")
        self.assertFalse(launcher.initialize_env(self.project))
        self.assertEqual(launcher.Config.from_project(self.project, {}).model, "custom-model")

    def test_long_command_provides_updates_and_preserves_captured_output(self):
        console = io.StringIO()
        with redirect_stdout(console):
            result = launcher.run_command(
                [sys.executable, "-c", "import time; time.sleep(0.15); print('finished')"],
                timeout=5, progress_message="Preparación en curso...", progress_interval=0.04,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("finished", result.stdout)
        self.assertIn("Preparación en curso", console.getvalue())

    def test_command_timeout_is_bounded_and_has_a_human_message(self):
        with self.assertRaisesRegex(launcher.LauncherError, "tiempo de espera"):
            launcher.run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)

    def test_dotenv_host_path_is_relative_to_project_and_environment_wins(self):
        (self.project / ".env").write_text(
            "OUTPUT_HOST_DIR='../Fotos listas'\n"
            "OUTPUT_BASE=/data/tiktok\n"
            "OLLAMA_BASE_URL=http://host.docker.internal:11434\n"
            "OLLAMA_MODEL=qwen2.5:3b # comentario\n",
            encoding="utf-8",
        )
        config = launcher.Config.from_project(self.project, {"OLLAMA_MODEL": "test-model"})
        self.assertEqual(config.output_dir, self.project.parent / "Fotos listas")
        self.assertEqual(config.ollama_url, "http://localhost:11434")
        self.assertEqual(config.model, "test-model")

    def test_dotenv_never_expands_or_executes_shell_values(self):
        envfile = self.project / ".env"
        envfile.write_text("GOOGLE_CHROME_PATH='$(whoami)'\n", encoding="utf-8")
        self.assertEqual(launcher.read_env(envfile, {})["GOOGLE_CHROME_PATH"], "$(whoami)")

    def test_quoted_dotenv_comment_is_removed_without_changing_literal_content(self):
        envfile = self.project / ".env"
        envfile.write_text(
            "OUTPUT_HOST_DIR='../Fotos # listas' # comentario fuera\n"
            'GOOGLE_CHROME_PATH="$(whoami) # literal" # no ejecutar\n'
            "TEST_URL=https://example.test/#section # comentario\n",
            encoding="utf-8",
        )
        values = launcher.read_env(envfile, {})
        self.assertEqual(values["GOOGLE_CHROME_PATH"], "$(whoami) # literal")
        self.assertEqual(values["TEST_URL"], "https://example.test/#section")
        self.assertEqual(launcher.Config.from_project(self.project, {}).output_dir, self.project.parent / "Fotos # listas")

    def test_unterminated_dotenv_quote_has_a_clear_error(self):
        envfile = self.project / ".env"
        envfile.write_text('OUTPUT_HOST_DIR="../Fotos\n', encoding="utf-8")
        with self.assertRaisesRegex(launcher.LauncherError, "OUTPUT_HOST_DIR.*comillas"):
            launcher.read_env(envfile, {})

    def test_documented_host_url_headless_and_port_configuration(self):
        config = launcher.Config.from_project(self.project, {
            "OLLAMA_HOST_URL": "http://localhost:12345",
            "OLLAMA_HOST_BASE_URL": "http://localhost:11111",
            "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
            "LENS_HEADLESS": "true",
            "TRANSLATOR_PORT": "8765",
        })
        self.assertEqual(config.ollama_url, "http://localhost:12345")
        self.assertEqual(config.api_url, "http://localhost:8765")
        self.assertTrue(config.headless)

    def test_tiktok_domain_boundary_and_credentials_are_validated(self):
        valid = [
            "https://vm.tiktok.com/ABC123/",
            "https://vt.tiktok.com/ABC123/",
            "https://www.tiktok.com/@user/photo/7659483315203804449",
        ]
        for url in valid:
            with self.subTest(url=url):
                self.assertEqual(launcher.validate_tiktok_url(url), url)
        invalid = [
            "https://tiktok.com.attacker.invalid/ABC/",
            "https://attacker-tiktok.com/ABC/",
            "https://tiktok.com@attacker.invalid/ABC/",
            "https://name:password@vm.tiktok.com/ABC/",
            "https://vm.tiktok.com:9999/ABC/",
            "file://vm.tiktok.com/ABC/",
            "https://vm.tiktok.com/",
        ]
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(launcher.LauncherError):
                launcher.validate_tiktok_url(url)

    def test_running_backend_is_reused_without_compose_or_build(self):
        self.seed_backend_state()
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ):
            launcher.ensure_backend(self.config, "docker")
            launcher.ensure_backend(self.config, "docker")
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(command[1:3] == ["image", "inspect"] for command in commands))

    def test_first_use_of_existing_image_does_not_rebuild(self):
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
        self.assertFalse(any("build" in command for command in commands))
        self.assertEqual(sum("up" in command for command in commands), 1)
        self.assertTrue(self.config.image_state_path.is_file())

    def test_missing_image_builds_once_then_reuses_running_backend(self):
        commands, fake_run = self.fake_docker(image_exists=False)
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
            launcher.ensure_backend(self.config, "docker")
        self.assertEqual(sum("build" in command for command in commands), 1)
        self.assertEqual(sum("up" in command for command in commands), 1)

    def test_dependency_upgrade_builds_once(self):
        self.seed_backend_state()
        (self.project / "translator/requirements.txt").write_text("fastapi==0.121.0\n", encoding="utf-8")
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
            launcher.ensure_backend(self.config, "docker")
        self.assertEqual(sum("build" in command for command in commands), 1)

    def test_source_change_restarts_once_without_rebuilding(self):
        self.seed_backend_state()
        (self.project / "translator/app.py").write_text("API_VERSION = 2\nNEW_FEATURE = True\n", encoding="utf-8")
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
            launcher.ensure_backend(self.config, "docker")
        self.assertFalse(any("build" in command for command in commands))
        self.assertEqual(sum("restart" in command for command in commands), 1)
        self.assertFalse(any("--force-recreate" in command for command in commands))

    def test_backend_environment_change_restarts_once_without_build(self):
        self.seed_backend_state()
        self.config.env["OCR_CPU_THREADS"] = "4"
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
            launcher.ensure_backend(self.config, "docker")
        self.assertFalse(any("build" in command for command in commands))
        self.assertEqual(sum("--force-recreate" in command for command in commands), 1)
        state = json.loads(self.config.image_state_path.read_text(encoding="utf-8"))
        self.assertNotIn("OCR_CPU_THREADS", state)

    def test_compose_change_restarts_without_building(self):
        self.seed_backend_state()
        (self.project / "docker-compose.yml").write_text("services: {}\nname: changed\n", encoding="utf-8")
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", return_value={"status": "ok", "api_version": 2}
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
        self.assertFalse(any("build" in command for command in commands))
        self.assertEqual(sum("--force-recreate" in command for command in commands), 1)

    def test_legacy_api_is_restarted_without_rebuilding_image(self):
        commands, fake_run = self.fake_docker()
        health = [{"status": "ok"}, {"status": "ok", "api_version": 2}]
        with patch.object(launcher, "run_command", side_effect=fake_run), patch.object(
            launcher, "optional_http_json", side_effect=health
        ), redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, "docker")
        self.assertEqual(sum("restart" in command for command in commands), 1)
        self.assertFalse(any("--force-recreate" in command for command in commands))
        self.assertFalse(any("build" in command for command in commands))

    def test_host_post_paths_cannot_escape_output_directory(self):
        result = {"output_dir": "/data/tiktok/7659483315203804449"}
        self.assertEqual(launcher.host_post_dir(self.config, result), self.config.output_dir / "7659483315203804449")
        for result in ({"post_id": "../outside"}, {"output_dir": "/data/tiktok/.."}):
            with self.assertRaises(launcher.LauncherError):
                launcher.host_post_dir(self.config, result)

    def test_provider_import_failure_keeps_all_slides_for_fallback(self):
        request = {"post_id": "123", "slides": [{"position": 1, "clean": "clean/01.jpg"}, {"position": 2, "clean": "clean/02.jpg"}]}
        with patch.object(launcher, "_google_lens_class", side_effect=ImportError("no playwright")), redirect_stdout(io.StringIO()):
            result = launcher.run_reverse_search(self.config, request)
        self.assertEqual([slide["position"] for slide in result], [1, 2])
        self.assertTrue(all(slide["status"] == "UNAVAILABLE" for slide in result))
        self.assertTrue(all(slide["candidates"] == [] for slide in result))

    def test_zero_interactive_timeout_disables_human_wait_without_disabling_provider(self):
        self.config.env["LENS_INTERACTIVE_TIMEOUT"] = "0"
        factory = Mock(return_value=Mock())
        launcher.run_reverse_search(self.config, {"post_id": "123", "slides": []}, provider_factory=factory)
        self.assertEqual(factory.call_args.kwargs["interactive_timeout"], 0)
        factory.return_value.close.assert_called_once()

    def test_unknown_configured_provider_reports_unavailable_without_import_or_upload(self):
        self.config.env['REVERSE_SEARCH_PROVIDER'] = 'unregistered_provider'
        request = {'post_id': '123', 'slides': [{'position': 1, 'clean': 'clean/01.jpg'}]}
        with patch.object(launcher, '_google_lens_class') as lens, redirect_stdout(io.StringIO()):
            results = launcher.run_reverse_search(self.config, request)
        lens.assert_not_called()
        self.assertEqual(results[0]['status'], 'UNAVAILABLE')
        self.assertEqual(results[0]['provider'], 'unregistered_provider')
        self.assertEqual(results[0]['candidates'], [])

    def test_other_launcher_timeouts_still_require_positive_values(self):
        for name in ("DOCKER_WAIT_SECONDS", "BACKEND_WAIT_SECONDS", "OLLAMA_WAIT_SECONDS", "JOB_TIMEOUT_SECONDS"):
            with self.subTest(name=name), self.assertRaises(launcher.LauncherError):
                launcher.Config.from_project(self.project, {name: "0"})

    def test_round_deadline_stops_later_slides_but_accepts_new_backend_budget(self):
        folder = self.config.output_dir/'123/clean'
        folder.mkdir(parents=True)
        for index in (1, 2):
            (folder/f'{index:02d}.jpg').write_bytes(b'fixture')
        clock = [10.0]
        provider = Mock()
        def search(*args, **kwargs):
            clock[0] = 12.0
            return {'status': 'NOT_FOUND', 'candidates': []}
        provider.search.side_effect = search
        session = {}
        slides = [{'position': index, 'clean': f'clean/{index:02d}.jpg',
                   'reverse_query': {'budget_seconds_remaining': 1}} for index in (1, 2)]
        request = {'post_id': '123', 'slides': slides}
        with patch.object(launcher.time, 'monotonic', side_effect=lambda: clock[0]), redirect_stdout(io.StringIO()):
            first = launcher.run_reverse_search(self.config, request, provider_factory=lambda **kw: provider, session=session)
            slides[0]['reverse_query']['budget_seconds_remaining'] = 300
            second = launcher.run_reverse_search(self.config, {'post_id': '123', 'slides': slides[:1]}, session=session)
        self.assertEqual([result['status'] for result in first], ['NOT_FOUND', 'TIMEOUT'])
        self.assertEqual(second[0]['status'], 'NOT_FOUND')
        self.assertEqual(provider.search.call_count, 2)
        self.assertEqual(session['deadline'], 312.0)
        provider.close.assert_not_called()

    def test_one_browser_search_failure_does_not_abort_later_slides(self):
        post = self.config.output_dir / "123" / "clean"
        post.mkdir(parents=True)
        (post / "01.jpg").write_bytes(b"fixture")
        (post / "02.jpg").write_bytes(b"fixture")
        request = {"output_dir": "/data/tiktok/123", "slides": [{"position": 1, "clean": "clean/01.jpg"}, {"position": 2, "clean": "clean/02.jpg"}]}

        class Provider:
            def __init__(self, **kwargs):
                self.calls = 0
                self.closed = False

            def search(self, path, force=False):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("Lens timeout")
                return {"status": "CANDIDATES", "candidates": [{"image_url": "https://example.test/photo.jpg"}]}

            def close(self):
                self.closed = True

        provider = Provider()
        with redirect_stdout(io.StringIO()):
            results = launcher.run_reverse_search(self.config, request, provider_factory=lambda **kw: provider)
        self.assertEqual([slide["status"] for slide in results], ["ERROR", "CANDIDATES"])
        self.assertEqual(provider.calls, 2)
        self.assertTrue(provider.closed)

    def test_clean_traversal_is_rejected_before_browser_upload(self):
        outside = self.config.output_dir / "outside.jpg"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"private")
        with self.assertRaises(launcher.LauncherError):
            launcher._clean_path(self.config.output_dir / "123", {"clean": "../outside.jpg"})

    def test_reverse_results_are_submitted_and_polling_finishes(self):
        submitted = []
        responses = iter([
            {"status": "WAITING_REVERSE_SEARCH", "reverse_search_request": {"post_id": "123", "slides": [{"position": 1, "clean": "clean/01.jpg"}]}},
            {"status": "COMPLETED", "result": {"post_id": "123", "cached": False}},
        ])

        def request(url, method="GET", body=None, **kwargs):
            if method == "POST":
                submitted.append((url, body))
                return {"job_id": "job-123"} if url.endswith("/jobs") else {"status": "ok"}
            return next(responses)

        with patch.object(launcher, "http_json", side_effect=request), patch.object(
            launcher, "_google_lens_class", side_effect=ImportError("no playwright")
        ), redirect_stdout(io.StringIO()):
            result = launcher.process_job(self.config, "https://vm.tiktok.com/ABC/")
        self.assertEqual(result["post_id"], "123")
        self.assertEqual(submitted[0][1]["reverse_search"], True)
        self.assertTrue(submitted[1][0].endswith("/jobs/job-123/reverse-search"))
        self.assertEqual(submitted[1][1]["slides"][0]["status"], "UNAVAILABLE")

    def test_later_reverse_round_is_submitted_after_backend_candidate_rejection(self):
        rounds = [{'state': 'WAITING_REVERSE_SEARCH', 'reverse_search_request': {
            'post_id': '123', 'round_id': round_id, 'slides': [{'position': 1, 'clean': 'clean/01.jpg'}]}}
            for round_id in ('clean-round', 'original-round')]
        replies = iter(rounds + [{'state': 'COMPLETED', 'result': {'status': 'OK', 'post_id': '123'}}])
        submitted = []
        def http(url, method='GET', body=None, **kwargs):
            if method=='POST':
                if url.endswith('/jobs'):
                    return {'job_id': 'job-123'}
                submitted.append(body)
                return {'accepted': True}
            return next(replies)
        with patch.object(launcher, 'http_json', side_effect=http), patch.object(
            launcher, 'run_reverse_search', return_value=[{'position': 1, 'status': 'NOT_FOUND'}]) as browser, redirect_stdout(io.StringIO()):
            result = launcher.process_job(self.config, 'https://vm.tiktok.com/ABC/')
        self.assertEqual(result['post_id'], '123')
        self.assertEqual(browser.call_count, 2)
        self.assertEqual([payload['round_id'] for payload in submitted], ['clean-round', 'original-round'])

    def test_backend_error_message_is_read_from_result(self):
        replies = [{"job_id": "abc"}, {"state": "ERROR", "result": {"status": "ERROR", "error": "Ollama no responde"}}]
        with patch.object(launcher, "http_json", side_effect=replies), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(launcher.LauncherError, "Ollama no responde"):
                launcher.process_job(self.config, "https://vm.tiktok.com/ABC/")

    def test_late_reverse_submission_does_not_hide_completed_backend_result(self):
        replies = iter([
            {"state": "WAITING_REVERSE_SEARCH", "reverse_search_request": {"post_id": "123", "slides": [{"position": 1, "clean": "clean/01.jpg"}]}},
            {"state": "COMPLETED", "result": {"status": "OK", "post_id": "123"}},
            {"state": "COMPLETED", "result": {"status": "OK", "post_id": "123"}},
        ])

        def request(url, method="GET", **kwargs):
            if method == "POST":
                if url.endswith("/jobs"):
                    return {"job_id": "abc"}
                raise launcher.LauncherError("El trabajo no está esperando candidatos", http_status=409)
            return next(replies)

        with patch.object(launcher, "http_json", side_effect=request), patch.object(
            launcher, "run_reverse_search", return_value=[{"position": 1, "status": "NOT_FOUND"}]
        ), redirect_stdout(io.StringIO()):
            result = launcher.process_job(self.config, "https://vm.tiktok.com/ABC/")
        self.assertEqual(result["post_id"], "123")

    def test_reverse_conflict_while_still_waiting_is_not_hidden(self):
        waiting = {"state": "WAITING_REVERSE_SEARCH", "reverse_search_request": {"post_id": "123", "slides": []}}

        def request(url, method="GET", **kwargs):
            if method == "POST":
                if url.endswith("/jobs"):
                    return {"job_id": "abc"}
                raise launcher.LauncherError("A genuine reverse conflict", http_status=409)
            return waiting

        with patch.object(launcher, "http_json", side_effect=request), patch.object(
            launcher, "run_reverse_search", return_value=[]
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(launcher.LauncherError, "genuine reverse conflict"):
                launcher.process_job(self.config, "https://vm.tiktok.com/ABC/")

    def test_failed_backend_message_survives_late_reverse_submission(self):
        replies = iter([
            {"state": "WAITING_REVERSE_SEARCH", "reverse_search_request": {"post_id": "123", "slides": []}},
            {"state": "ERROR", "result": {"status": "ERROR", "error": "Ollama is unavailable"}},
            {"state": "ERROR", "result": {"status": "ERROR", "error": "Ollama is unavailable"}},
        ])

        def request(url, method="GET", **kwargs):
            if method == "POST":
                if url.endswith("/jobs"):
                    return {"job_id": "abc"}
                raise launcher.LauncherError("Late reverse submission", http_status=409)
            return next(replies)

        with patch.object(launcher, "http_json", side_effect=request), patch.object(
            launcher, "run_reverse_search", return_value=[]
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(launcher.LauncherError, "Ollama is unavailable"):
                launcher.process_job(self.config, "https://vm.tiktok.com/ABC/")

    def test_custom_image_configuration_and_compose_share_the_same_variable(self):
        config = launcher.Config.from_project(self.project, {"TRANSLATOR_IMAGE": "custom-translator:v2"})
        self.assertEqual(config.image, "custom-translator:v2")
        actual_compose = Path(launcher.__file__).resolve().parents[1] / "docker-compose.yml"
        self.assertIn("image: ${TRANSLATOR_IMAGE:-tiktoktranslator-translator:latest}", actual_compose.read_text(encoding="utf-8"))

    def test_large_browser_error_and_candidate_urls_still_allow_fallback(self):
        result = launcher._reverse_payload({
            "status": "UNAVAILABLE", "error": "x" * 5000,
            "candidates": [{"image_url": "https://example.test/" + "y" * 5000}],
        }, 1, 2)
        self.assertEqual(len(result["error"]), 2000)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["status"], "UNAVAILABLE")

    def test_skipped_no_images_is_reported_as_unsupported(self):
        output = io.StringIO()
        with redirect_stdout(output):
            launcher.report_result(self.config, {
                "status": "SKIPPED_NO_IMAGES", "error": "Los vídeos no están soportados.",
                "post_id": "123", "output_dir": "/data/tiktok/123",
            }, no_open=True)
        self.assertIn("vídeos no están soportados", output.getvalue())
        self.assertNotIn("Terminado", output.getvalue())

    def test_prepared_browser_is_not_installed_again(self):
        executable = self.project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"fixture")
        commands = []

        def run(args, **kwargs):
            commands.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch.object(launcher, "run_command", side_effect=run):
            launcher.prepare_browser_runtime(self.config)
            launcher.prepare_browser_runtime(self.config)
        self.assertFalse(any("pip" in command or "install" in command for command in commands))

    def test_browser_download_failure_still_returns_prepared_interpreter(self):
        executable = self.project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"fixture")

        def run(args, **kwargs):
            if "install" in args:
                raise launcher.LauncherError("Download unavailable")
            ready = args[-1] == "import playwright"
            return subprocess.CompletedProcess(args, 0 if ready else 1, "", "")

        with patch.object(launcher, "run_command", side_effect=run), patch.object(
            launcher, "_known_program", return_value=None
        ), redirect_stdout(io.StringIO()):
            interpreter = launcher.prepare_browser_runtime(self.config)
        self.assertEqual(interpreter, executable.resolve())

    def test_installed_chrome_does_not_trigger_an_extra_browser_download(self):
        executable = self.project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"fixture")
        commands = []

        def run(args, **kwargs):
            commands.append(args)
            return subprocess.CompletedProcess(args, 0 if args[-1] == "import playwright" else 1, "", "")

        with patch.object(launcher, "run_command", side_effect=run), patch.object(
            launcher, "_known_program", return_value="C:/Chrome/chrome.exe"
        ):
            launcher.prepare_browser_runtime(self.config)
        self.assertFalse(any("install" in command for command in commands))

    def test_model_check_reports_missing_model_without_changing_default(self):
        with patch.object(launcher, "optional_http_json", return_value={"models": [{"name": "other-model:latest"}]}):
            with self.assertRaisesRegex(launcher.LauncherError, "ollama pull gemma3:4b"):
                launcher.ensure_ollama(self.config)

    def test_default_translation_model_reuses_primary_and_queries_tags_once(self):
        self.assertIsNone(self.config.translation_model)
        self.assertEqual(self.config.model, "gemma3:4b")
        with patch.object(launcher, "optional_http_json", return_value={"models": [{"name": "gemma3:4b"}]}) as tags:
            launcher.ensure_ollama(self.config)
        tags.assert_called_once_with(self.config.ollama_url + "/api/tags")

    def test_two_configured_models_are_verified_in_one_tags_request(self):
        config = launcher.Config.from_project(self.project, {
            "OLLAMA_MODEL": "qwen2.5:3b", "OLLAMA_TRANSLATION_MODEL": "translategemma:4b",
        })
        with patch.object(launcher, "optional_http_json", return_value={"models": [
            {"name": "qwen2.5:3b"}, {"name": "translategemma:4b"},
        ]}) as tags:
            launcher.ensure_ollama(config)
        tags.assert_called_once_with(config.ollama_url + "/api/tags")
        self.assertEqual(config.translation_model, "translategemma:4b")

    def test_missing_translation_model_reports_its_own_install_command(self):
        config = launcher.Config.from_project(self.project, {"OLLAMA_TRANSLATION_MODEL": "translategemma:4b"})
        with patch.object(launcher, "optional_http_json", return_value={"models": [{"name": "gemma3:4b"}]}) as tags:
            with self.assertRaisesRegex(launcher.LauncherError, "ollama pull translategemma:4b"):
                launcher.ensure_ollama(config)
        tags.assert_called_once_with(config.ollama_url + "/api/tags")

    def test_empty_translation_model_keeps_single_model_workflow(self):
        config = launcher.Config.from_project(self.project, {"OLLAMA_TRANSLATION_MODEL": "  "})
        self.assertIsNone(config.translation_model)
        with patch.object(launcher, "optional_http_json", return_value={"models": [{"name": "gemma3:4b"}]}):
            launcher.ensure_ollama(config)

    def test_translation_model_changes_configuration_but_not_dependency_fingerprint(self):
        config = launcher.Config.from_project(self.project, {"OLLAMA_TRANSLATION_MODEL": "translategemma:4b"})
        self.assertNotEqual(launcher.configuration_fingerprint(self.config), launcher.configuration_fingerprint(config))
        self.assertEqual(launcher.dependency_fingerprint(self.config.project_dir), launcher.dependency_fingerprint(config.project_dir))

    def test_interactive_launcher_processes_two_urls_with_one_startup_and_browser(self):
        urls = ['https://vm.tiktok.com/FIRST/', 'https://vm.tiktok.com/SECOND/']
        provider = Mock()
        sessions = []

        def process(config, url, **kwargs):
            session = kwargs['reverse_session']
            sessions.append(session)
            session.setdefault('provider', provider)
            return {'status': 'OK', 'post_id': '123', 'launcher_elapsed_seconds': 0.1}

        with patch.object(launcher, 'initialize_env', return_value=False), patch.object(
            launcher.Config, 'from_project', return_value=self.config
        ), patch.object(launcher, 'setup_logging', return_value=self.project/'launcher.log'), patch.object(
            launcher, 'ensure_docker', return_value='docker'
        ) as docker, patch.object(launcher, 'ensure_ollama') as ollama, patch.object(
            launcher, 'ensure_backend'
        ) as backend, patch.object(launcher, 'process_job', side_effect=process) as jobs, patch.dict(
            os.environ, {'TIKTOK_LAUNCHER_BOOTSTRAPPED': '1'}
        ), patch('builtins.input', side_effect=urls+['SALIR']) as prompt, redirect_stdout(io.StringIO()) as console:
            status = launcher.main(['--no-open'])

        self.assertEqual(status, 0)
        docker.assert_called_once_with(self.config)
        ollama.assert_called_once_with(self.config)
        backend.assert_called_once_with(self.config, 'docker')
        self.assertEqual([call.args[1] for call in jobs.call_args_list], urls)
        self.assertIs(sessions[0], sessions[1])
        provider.close.assert_called_once()
        self.assertIn('Introduce otro enlace', prompt.call_args_list[1].args[0])
        self.assertIn('Completado.', console.getvalue())

    def test_interactive_launcher_accepts_exit_aliases_without_starting_services(self):
        for command in ['salir', 'exit', 'q']:
            with self.subTest(command=command), patch.object(launcher, 'initialize_env', return_value=False), patch.object(
                launcher.Config, 'from_project', return_value=self.config
            ), patch.object(launcher, 'setup_logging', return_value=self.project/'launcher.log'), patch.object(
                launcher, 'ensure_docker'
            ) as docker, patch('builtins.input', return_value=command), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main([]), 0)
            docker.assert_not_called()

    def test_bad_url_and_eof_leave_prompt_usable(self):
        with patch('builtins.input', side_effect=['not-a-url', 'https://vm.tiktok.com/NEXT/']), redirect_stdout(io.StringIO()) as console:
            self.assertEqual(launcher._prompt_url(), 'https://vm.tiktok.com/NEXT/')
        self.assertIn('Enlace no válido', console.getvalue())
        with patch('builtins.input', side_effect=EOFError):
            self.assertIsNone(launcher._prompt_url(following=True))

    def test_shared_browser_is_not_closed_after_each_job_and_deadline_resets(self):
        provider = Mock()
        session = {'provider': provider, 'deadline': 10}
        with patch.object(launcher, '_process_job', return_value={'post_id': '123'}) as job:
            launcher.process_job(self.config, 'https://vm.tiktok.com/FIRST/', reverse_session=session)
            session['deadline'] = 20
            launcher.process_job(self.config, 'https://vm.tiktok.com/SECOND/', reverse_session=session)
        provider.close.assert_not_called()
        self.assertNotIn('deadline', session)
        self.assertEqual(provider.reset_for_post.call_count, 2)
        self.assertIs(job.call_args.kwargs['reverse_session'], session)
        launcher._close_reverse_session(session)
        provider.close.assert_called_once()

    def test_incomplete_result_reports_each_missing_output(self):
        with redirect_stdout(io.StringIO()) as console:
            launcher.report_result(self.config, {'status': 'INCOMPLETE', 'post_id': '123',
                'validation': {'missing': ['recovered/03_SOURCE.*', 'traducción diapositiva 3']}}, no_open=True)
        self.assertIn('INCOMPLETE', console.getvalue())
        self.assertIn('Falta: recovered/03_SOURCE.*', console.getvalue())
        self.assertIn('Falta: traducción diapositiva 3', console.getvalue())

    def test_web_candidate_download_details_survive_launcher_payload(self):
        bridge = {'data_base64': 'eHl6', 'url': 'https://example.test/real.png',
            'http_status': 200, 'content_type': 'image/png', 'bytes': 3}
        payload = launcher._reverse_payload({'status': 'CANDIDATES', 'candidates': [{
            'image_url': 'https://example.test/proxy', 'source_url': 'https://example.test/page',
            'alternate_urls': ['https://example.test/real.png'], 'width': 1920, 'height': 1080,
            'thumbnail': True, 'browser_download': bridge}]}, 1, 0.1)
        candidate = payload['candidates'][0]
        self.assertEqual(candidate['alternate_urls'], ['https://example.test/real.png'])
        self.assertEqual(candidate['width'], 1920)
        self.assertEqual(candidate['height'], 1080)
        self.assertTrue(candidate['thumbnail'])
        self.assertEqual(candidate['browser_download'], bridge)

    def test_verified_runtime_dependencies_avoid_build_and_preserve_image_fingerprint(self):
        self.seed_backend_state()
        image_dependencies = launcher.dependency_fingerprint(self.project)
        (self.project/'translator/requirements.txt').write_text('fastapi==0.120.0\nctranslate2==4.8.2\n', encoding='utf-8')
        (self.project/'translator/app.py').write_text('API_VERSION = 2\nSOURCE_CHANGED = True\n', encoding='utf-8')
        commands, fake_run = self.fake_docker()
        with patch.object(launcher, 'run_command', side_effect=fake_run), patch.object(
            launcher, 'optional_http_json', return_value={'status': 'ok', 'api_version': 2}
        ), patch.object(launcher, 'runtime_dependencies_match', return_value=True) as probe, redirect_stdout(io.StringIO()):
            launcher.ensure_backend(self.config, 'docker')
            launcher.ensure_backend(self.config, 'docker')
        self.assertFalse(any('build' in command for command in commands))
        self.assertFalse(any('--force-recreate' in command for command in commands))
        self.assertEqual(sum('restart' in command for command in commands), 1)
        self.assertEqual(probe.call_count, 2)
        state = json.loads(self.config.image_state_path.read_text(encoding='utf-8'))
        self.assertEqual(state['dependencies'], image_dependencies)
        self.assertEqual(state['runtime_dependencies'], launcher.dependency_fingerprint(self.project))

    def test_runtime_dependency_probe_requires_explicit_verified_response(self):
        for response, expected in [('invalid', False), ('{}', False), ('{"matches": false}', False), ('{"matches": true}', True)]:
            with self.subTest(response=response), patch.object(launcher, 'run_command', return_value=
                subprocess.CompletedProcess(['docker'], 0, response, '')) as command:
                self.assertEqual(launcher.runtime_dependencies_match(self.config, 'docker'), expected)
            self.assertEqual(command.call_args.args[0][:4], ['docker', 'exec', 'tiktok-translator', 'python'])


if __name__ == "__main__":
    unittest.main()
