"""Check the existing Windows runtime; large downloads require explicit flags."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launcher.main import (
    Config,
    LauncherError,
    _compatible_health,
    _known_program,
    _program_files,
    _read_json_file,
    _write_json_file,
    dependency_fingerprint,
    ensure_backend,
    ensure_docker,
    ensure_ollama,
    initialize_env,
    optional_http_json,
    run_command,
)


def browser_available(config: Config) -> bool:
    configured = config.env.get("GOOGLE_CHROME_PATH")
    if configured and Path(configured).is_file():
        return True
    paths = [base / "Google/Chrome/Application/chrome.exe" for base in _program_files()]
    paths.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe")
    if not configured and _known_program(paths, "chrome"):
        return True
    if importlib.util.find_spec("playwright") is None:
        return False
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path).is_file()


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-models", action="store_true")
    parser.add_argument("--install-browser", action="store_true")
    parser.add_argument("--build-backend", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only and (args.pull_models or args.install_browser or args.build_backend):
        parser.error("--check-only no admite opciones de descarga o instalación")
    problems: list[str] = []
    if sys.version_info < (3, 11):
        print("Se necesita Python 3.11 o posterior.")
        return 1
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if not (ROOT / ".env").is_file():
        if args.check_only:
            print("Falta .env. Ejecuta .\\setup.ps1 para crear la configuración local.")
            return 1
        initialize_env(ROOT)
        print(".env creado desde .env.example; no se reemplazan configuraciones existentes.")
    config = Config.from_project(ROOT)
    if not args.check_only:
        for directory in (config.output_dir, ROOT / "secrets", ROOT / ".cache", ROOT / "logs"):
            directory.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("playwright") is None:
        problems.append("Falta Playwright: ejecuta .\\setup.ps1.")
    else:
        print("Playwright: instalado.")
        if not browser_available(config):
            if args.install_browser:
                print("Descarga solicitada: Chromium de Playwright (cientos de MB).")
                run_command([sys.executable, "-m", "playwright", "install", "chromium"],
                            cwd=ROOT, timeout=900, progress_message="La descarga de Chromium continúa...")
            else:
                problems.append("Falta Chrome/Chromium. Instala Chrome o ejecuta .\\setup.ps1 -InstallBrowser.")
        if browser_available(config):
            print("Navegador: Chrome o Chromium disponible.")
    docker: str | None = None
    if args.check_only:
        docker = shutil.which("docker")
        if not docker:
            problems.append("Docker no está en PATH. Instala Docker Desktop con contenedores Linux.")
        else:
            ready = run_command([docker, "info", "--format", "{{.OSType}}"], timeout=15, check=False)
            if ready.returncode or ready.stdout.strip() != "linux":
                problems.append("Docker no responde con contenedores Linux. Abre Docker Desktop.")
                docker = None
    else:
        try:
            docker = ensure_docker(config)
            operating_system = run_command([docker, "info", "--format", "{{.OSType}}"], timeout=15)
            if operating_system.stdout.strip() != "linux":
                problems.append("Configura Docker Desktop para usar contenedores Linux.")
                docker = None
        except LauncherError as exc:
            problems.append(str(exc))
    needs_build = False
    if docker:
        compose = run_command([docker, "compose", "version"], timeout=15, check=False)
        if compose.returncode:
            problems.append("Falta Docker Compose v2 o posterior.")
        else:
            print("Docker: disponible; " + compose.stdout.strip())
        image = run_command([docker, "image", "inspect", config.image], timeout=15, check=False)
        state = _read_json_file(config.image_state_path)
        dependencies = dependency_fingerprint(ROOT)
        needs_build = image.returncode != 0 or bool(state.get("dependencies") and state["dependencies"] != dependencies)
        if needs_build and not args.build_backend:
            problems.append("La imagen del backend necesita preparación: .\\setup.ps1 -BuildBackend. "
                            "El primer build descarga varios GB y modelos OCR al iniciar.")
    if not args.check_only:
        try:
            ensure_ollama(config)
        except LauncherError as exc:
            # Starting a stopped local Ollama server is handled by the launcher.
            if optional_http_json(config.ollama_url + "/api/tags") is None:
                problems.append(str(exc))
    tags = optional_http_json(config.ollama_url + "/api/tags")
    if tags is None:
        problems.append("Ollama no responde. Instala e inicia Ollama para Windows.")
    else:
        available = {str(item.get("name") or item.get("model")) for item in tags.get("models", [])}
        for model in dict.fromkeys((config.model, config.translation_model or config.model)):
            aliases = {model, model + ":latest"} if ":" not in model else {model}
            if available.intersection(aliases):
                print(f"Modelo: {model}, instalado.")
                continue
            if not args.pull_models:
                problems.append(f"Falta {model}: ollama pull {model}, o .\\setup.ps1 -PullModels "
                                "(descarga de varios GB por modelo).")
                continue
            local_app = Path(os.environ.get("LOCALAPPDATA", ""))
            ollama = _known_program([local_app / "Programs/Ollama/ollama.exe",
                                     *[base / "Ollama/ollama.exe" for base in _program_files()]], "ollama")
            if not ollama:
                problems.append("Ollama responde, pero no se ha localizado su CLI para descargar el modelo.")
                continue
            print(f"Descarga solicitada: {model}. Puede ocupar varios GB y tardar varios minutos.")
            run_command([ollama, "pull", model], cwd=ROOT, timeout=3600,
                        progress_message=f"La descarga de {model} continúa...")
    if args.check_only:
        if not _compatible_health(optional_http_json(config.api_url + "/health")):
            problems.append("El backend no responde o necesita actualizarse. Ejecuta .\\setup.ps1.")
    if problems:
        print("\nPreparación pendiente:")
        for problem in dict.fromkeys(problems):
            print("- " + problem)
        return 1
    if not args.check_only:
        if needs_build:
            print("Build solicitado: imagen Docker y modelos OCR. Se reutilizan en ejecuciones posteriores.")
        ensure_backend(config, docker)
        _write_json_file(ROOT / ".cache" / "launcher-host-dependencies.json", {
            "requirements": hashlib.sha256((ROOT / "requirements-launcher.txt").read_bytes()).hexdigest()
        })
    print("\nPreparado. Ejecuta .\\run.ps1 y pega el enlace de TikTok.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, OSError, ValueError) as exc:
        print(f"Setup incompleto: {exc}")
        raise SystemExit(1)
