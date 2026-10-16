"""Local Windows entry point; Docker and HTTP orchestration uses only stdlib.

Playwright is imported only by the optional host reverse-search provider. The
backend keeps a single OCR model alive, and this launcher reuses its container.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


LOG = logging.getLogger("tiktok.launcher")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False
API_VERSION = 2


class LauncherError(RuntimeError):
    """An error that can be shown directly to the person using the launcher."""

    def __init__(self, message: str, *, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


def read_env(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Read simple dotenv values without executing or expanding any shell text."""
    values: dict[str, str] = {}
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            if value[:1] in {"\"", "'"}:
                quoted = re.fullmatch(r"([\"'])(.*?)\1\s*(?:#.*)?", value)
                if not quoted:
                    raise LauncherError(f"El valor de {key} en .env tiene comillas sin cerrar o texto fuera de ellas.")
                value = quoted.group(2)
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            values[key] = value
    values.update(os.environ if environ is None else environ)
    return values


def initialize_env(project_dir: Path) -> bool:
    """Create the local configuration on first use, preserving existing values."""
    destination = project_dir / ".env"
    if destination.exists():
        return False
    example = project_dir / ".env.example"
    if not example.is_file():
        raise LauncherError("Falta .env.example; no se puede preparar la configuración local.")
    shutil.copyfile(example, destination)
    return True


def _number(env: dict[str, str], name: str, default: float, *, allow_zero: bool = False) -> float:
    requirement = "un número mayor o igual que cero" if allow_zero else "un número positivo"
    try:
        result = float(env.get(name, str(default)))
    except ValueError as exc:
        raise LauncherError(f"{name} debe ser {requirement} en .env.") from exc
    if not (0 <= result < float("inf") and (allow_zero or result > 0)):
        raise LauncherError(f"{name} debe ser {requirement} en .env.")
    return result


def _host_ollama_url(env: dict[str, str]) -> str:
    base = env.get("OLLAMA_HOST_URL") or env.get("OLLAMA_HOST_BASE_URL") or env.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LauncherError("OLLAMA_BASE_URL debe ser una URL HTTP válida.")
    if parsed.hostname == "host.docker.internal":
        port = f":{parsed.port}" if parsed.port else ""
        parsed = parsed._replace(netloc="localhost" + port)
    return urlunsplit(parsed).rstrip("/")


@dataclass
class Config:
    project_dir: Path
    output_dir: Path
    env: dict[str, str] = field(repr=False)
    api_url: str = "http://localhost:8000"
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma3:4b"
    translation_model: str | None = None
    image: str = "tiktoktranslator-translator:latest"
    docker_timeout: float = 180
    backend_timeout: float = 120
    ollama_timeout: float = 60
    job_timeout: float = 3600
    poll_interval: float = 1
    headless: bool = False

    @classmethod
    def from_project(
        cls, project_dir: Path, environ: dict[str, str] | None = None
    ) -> "Config":
        root = project_dir.resolve()
        env = read_env(root / ".env", environ)
        configured_output = env.get("OUTPUT_HOST_DIR", "").strip()
        output = Path(configured_output) if configured_output else root.parent / "TikTokTraducidos"
        if not output.is_absolute():
            output = root / output
        port = env.get("TRANSLATOR_PORT", "8000")
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise LauncherError("TRANSLATOR_PORT debe estar entre 1 y 65535.")
        return cls(
            project_dir=root,
            output_dir=output.resolve(),
            env=env,
            api_url=env.get("TRANSLATOR_BASE_URL", f"http://localhost:{port}").rstrip("/"),
            ollama_url=_host_ollama_url(env),
            model=env.get("OLLAMA_MODEL", "gemma3:4b"),
            translation_model=env.get("OLLAMA_TRANSLATION_MODEL", "").strip() or None,
            image=env.get("TRANSLATOR_IMAGE", "tiktoktranslator-translator:latest"),
            docker_timeout=_number(env, "DOCKER_WAIT_SECONDS", 180),
            backend_timeout=_number(env, "BACKEND_WAIT_SECONDS", 120),
            ollama_timeout=_number(env, "OLLAMA_WAIT_SECONDS", 60),
            job_timeout=_number(env, "JOB_TIMEOUT_SECONDS", 3600),
            poll_interval=_number(env, "JOB_POLL_SECONDS", 1),
            headless=env.get("LENS_HEADLESS", env.get("HEADLESS", "false")).lower() in {"true", "1", "yes"},
        )

    @property
    def command_env(self) -> dict[str, str]:
        values = dict(os.environ)
        values.update(self.env)
        # An absolute host path also makes compose independent of the user's cwd.
        values["OUTPUT_HOST_DIR"] = str(self.output_dir)
        return values

    @property
    def image_state_path(self) -> Path:
        return self.project_dir / ".cache" / "launcher-image.json"


def setup_logging(project_dir: Path) -> Path:
    path = project_dir / "logs" / "launcher.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(handler)
    # Direct script execution and package imports have different logger names.
    # Keep provider tracebacks in the same file; show only actionable warnings.
    for name in ("reverse_search", "launcher.reverse_search"):
        provider_log = logging.getLogger(name)
        provider_log.setLevel(logging.DEBUG)
        provider_log.propagate = False
        provider_log.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("%(message)s"))
        provider_log.addHandler(console)
    return path


def validate_tiktok_url(value: str) -> str:
    value = value.strip()
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:
        raise LauncherError("El enlace de TikTok no es válido.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not (host == "tiktok.com" or host.endswith(".tiktok.com"))
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 80, 443}
        or parsed.path in {"", "/"}
        or any(char.isspace() for char in value)
    ):
        raise LauncherError("Pega un enlace de un post de TikTok (tiktok.com, vm.tiktok.com o vt.tiktok.com).")
    return value


def http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 15,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = str(json.loads(detail).get("detail", detail))
        except (ValueError, AttributeError):
            pass
        raise LauncherError(f"El servicio respondió HTTP {exc.code}: {detail[:500]}", http_status=exc.code) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise LauncherError(f"No se puede conectar con {url}: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise LauncherError(f"Respuesta JSON no válida de {url}.") from exc
    if not isinstance(payload, dict):
        raise LauncherError(f"Respuesta inesperada de {url}.")
    return payload


def optional_http_json(url: str, timeout: float = 3) -> dict[str, Any] | None:
    try:
        return http_json(url, timeout=timeout)
    except LauncherError:
        LOG.debug("Servicio todavía no disponible: %s", url, exc_info=True)
        return None


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60,
    check: bool = True,
    progress_message: str | None = None,
    progress_interval: float = 20,
) -> subprocess.CompletedProcess[str]:
    LOG.debug("Comando: %r (cwd=%s)", args, cwd)
    process = None
    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            interval = min(progress_interval, remaining) if progress_message else remaining
            try:
                stdout, stderr = process.communicate(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                if progress_message:
                    print(progress_message, flush=True)
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    except FileNotFoundError as exc:
        raise LauncherError(f"No se ha encontrado el programa: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            process.kill()
            stdout, stderr = process.communicate()
            LOG.debug("Salida antes del timeout: %s\n%s", stdout, stderr)
        raise LauncherError(f"El comando {Path(args[0]).name} superó el tiempo de espera ({timeout:g} s).") from exc
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        raise
    LOG.debug("Salida (%s): %s\n%s", result.returncode, result.stdout, result.stderr)
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise LauncherError(f"Falló {Path(args[0]).name}: {detail[-1000:]}")
    return result


def _known_program(paths: list[Path], command: str | None = None) -> str | None:
    found = shutil.which(command) if command else None
    if found:
        return found
    return next((str(path) for path in paths if path.is_file()), None)


def _program_files() -> list[Path]:
    return [Path(os.environ.get("ProgramFiles", "C:/Program Files")), Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))]


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def start_hidden_application(executable: str, arguments: tuple[str, ...] = ()) -> None:
    if os.name != "nt":
        subprocess.Popen([executable, *arguments], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    script = "Start-Process -FilePath " + _powershell_quote(executable) + " -WindowStyle Hidden"
    if arguments:
        script += " -ArgumentList " + ",".join(_powershell_quote(arg) for arg in arguments)
    run_command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout=15)


def _wait_for(check: Callable[[], Any], timeout: float, description: str) -> Any:
    deadline = time.monotonic() + timeout
    last_notice = time.monotonic()
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        if time.monotonic() - last_notice >= 20:
            print(f"{description}...", flush=True)
            last_notice = time.monotonic()
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2, remaining))
    raise LauncherError(f"{description}: se agotó el tiempo de espera ({timeout:g} s).")


def ensure_docker(config: Config) -> str:
    paths = [base / "Docker/Docker/resources/bin/docker.exe" for base in _program_files()]
    local_app = Path(os.environ.get("LOCALAPPDATA", "C:/Users/Default/AppData/Local"))
    paths.append(local_app / "Programs/DockerDesktop/resources/bin/docker.exe")
    docker = _known_program(paths, "docker")
    if not docker:
        raise LauncherError("No se ha encontrado Docker Desktop. Instálalo antes de usar el traductor.")

    def ready() -> bool:
        try:
            return run_command([docker, "info", "--format", "{{.ServerVersion}}"], timeout=8, check=False).returncode == 0
        except LauncherError:
            return False

    if ready():
        return docker
    desktop_paths = [base / "Docker/Docker/Docker Desktop.exe" for base in _program_files()]
    desktop_paths.append(local_app / "Programs/DockerDesktop/Docker Desktop.exe")
    if config.env.get("DOCKER_DESKTOP_PATH"):
        desktop_paths.insert(0, Path(config.env["DOCKER_DESKTOP_PATH"]))
    desktop = _known_program(desktop_paths)
    if not desktop:
        raise LauncherError("Docker no responde y no se ha encontrado Docker Desktop para iniciarlo.")
    print("Iniciando Docker Desktop; el primer arranque puede tardar un poco...", flush=True)
    start_hidden_application(desktop)
    _wait_for(ready, config.docker_timeout, "Esperando a Docker Desktop")
    return docker


def ensure_ollama(config: Config) -> None:
    tags = optional_http_json(config.ollama_url + "/api/tags")
    if tags is None:
        host = urlsplit(config.ollama_url).hostname
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise LauncherError(f"Ollama no responde en {config.ollama_url}.")
        local_app = Path(os.environ.get("LOCALAPPDATA", "C:/Users/Default/AppData/Local"))
        paths = [local_app / "Programs/Ollama/ollama.exe", *[base / "Ollama/ollama.exe" for base in _program_files()]]
        ollama = _known_program(paths, "ollama")
        if not ollama:
            raise LauncherError("Ollama no está iniciado y no se ha encontrado una instalación local.")
        print("Iniciando Ollama...", flush=True)
        start_hidden_application(ollama, ("serve",))
        tags = _wait_for(lambda: optional_http_json(config.ollama_url + "/api/tags"), config.ollama_timeout, "Esperando a Ollama")
    available = {str(item.get("name") or item.get("model")) for item in tags.get("models", []) if isinstance(item, dict)}
    required_models = dict.fromkeys((config.model, config.translation_model or config.model))
    for model in required_models:
        names = {model, model + ":latest"} if ":" not in model else {model}
        if not available.intersection(names):
            raise LauncherError(f"El modelo {model} no está instalado en Ollama. Instálalo con: ollama pull {model}")


def dependency_fingerprint(project_dir: Path) -> str:
    digest = hashlib.sha256()
    for relative in ("translator/Dockerfile", "translator/requirements.txt"):
        path = project_dir / relative
        if not path.is_file():
            raise LauncherError(f"Falta un archivo del proyecto: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_fingerprint(project_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((project_dir / "translator").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(project_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configuration_fingerprint(config: Config) -> str:
    """Hash the effective compose configuration without storing any env secrets."""
    compose = (config.project_dir / "docker-compose.yml").read_bytes()
    keys = set(read_env(config.project_dir / ".env", {}))
    keys.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", compose.decode("utf-8-sig")))
    keys.update({
        "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_TRANSLATION_MODEL", "OLLAMA_KEEP_ALIVE", "OCR_CPU_THREADS",
        "REVERSE_SEARCH_WAIT_SECONDS", "VISUAL_MATCH_THRESHOLD", "CANDIDATE_MIN_DIMENSION",
        "LENS_MAX_CANDIDATES", "OUTPUT_HOST_DIR", "TIKTOK_COOKIES", "TRANSLATOR_PORT",
        "TRANSLATOR_IMAGE", "TZ",
    })
    environment = config.command_env
    digest = hashlib.sha256(compose)
    digest.update(json.dumps({key: environment.get(key, "") for key in sorted(keys)}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_backend_state(config: Config, dependencies: str, source: str, *, runtime_dependencies: str | None = None) -> None:
    state = {
        "image": config.image,
        "dependencies": dependencies,
        "source": source,
        "output_dir": str(config.output_dir),
        "project_dir": str(config.project_dir),
        "api_version": API_VERSION,
        "configuration": configuration_fingerprint(config),
    }
    if runtime_dependencies:
        # An installed runtime package does not alter the existing Docker image.
        state["runtime_dependencies"] = runtime_dependencies
    _write_json_file(config.image_state_path, state)


def _compatible_health(value: dict[str, Any] | None) -> bool:
    return bool(value and value.get("api_version") == API_VERSION and value.get("status") in {"ok", "healthy", "ready"})


def runtime_dependencies_match(config: Config, docker: str) -> bool:
    """Verify requirements in the running container without installing anything."""
    requirements = [line.strip() for line in (config.project_dir / "translator/requirements.txt").read_text(
        encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    code = """import importlib.metadata as metadata, json, sys
try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
missing = []
for value in json.loads(sys.argv[1]):
    requirement = Requirement(value)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    try:
        version = metadata.version(requirement.name)
        if version not in requirement.specifier:
            missing.append(str(requirement))
    except metadata.PackageNotFoundError:
        missing.append(str(requirement))
if not missing:
    names = {Requirement(value).name.lower() for value in json.loads(sys.argv[1])}
    if 'ctranslate2' in names:
        import ctranslate2
    if 'sentencepiece' in names:
        import sentencepiece
print(json.dumps({'matches': not missing, 'missing': missing}))
"""
    try:
        result = run_command([docker, "exec", "tiktok-translator", "python", "-c", code,
                              json.dumps(requirements)], cwd=config.project_dir, timeout=30, check=False)
        probe = json.loads(result.stdout) if result.returncode == 0 else {}
        matches = isinstance(probe, dict) and probe.get("matches") is True
        LOG.info("Dependencias del contenedor activo: %r", probe)
        return matches
    except (LauncherError, ValueError):
        LOG.debug("No se pudieron verificar las dependencias del contenedor activo.", exc_info=True)
        return False


def ensure_backend(config: Config, docker: str) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = dependency_fingerprint(config.project_dir)
    source = source_fingerprint(config.project_dir)
    configuration = configuration_fingerprint(config)
    state = _read_json_file(config.image_state_path)
    image_exists = run_command([docker, "image", "inspect", config.image], cwd=config.project_dir, env=config.command_env, timeout=15, check=False).returncode == 0
    dependency_changed = bool(state.get("dependencies") and state["dependencies"] != dependencies)
    needs_build = not image_exists or dependency_changed
    health = optional_http_json(config.api_url + "/health")
    configuration_changed = bool(state.get("configuration") and state["configuration"] != configuration)
    runtime_satisfied = bool(image_exists and dependency_changed and state.get("image") == config.image
                             and not configuration_changed
                             and runtime_dependencies_match(config, docker))
    if runtime_satisfied:
        needs_build = False
        LOG.info("Reutilizando las dependencias verificadas del contenedor; la imagen no se reconstruye.")
    same_configuration = (
        state.get("image") == config.image
        and (state.get("dependencies") == dependencies or runtime_satisfied)
        and state.get("source") == source
        and state.get("output_dir") == str(config.output_dir)
        and state.get("project_dir") == str(config.project_dir)
        and state.get("configuration") == configuration
    )
    if not needs_build and same_configuration and _compatible_health(health):
        LOG.info("Reutilizando el servicio translator ya iniciado.")
        if runtime_satisfied:
            save_backend_state(config, state["dependencies"], source, runtime_dependencies=dependencies)
        return
    compose = [docker, "compose", "--project-directory", str(config.project_dir), "-f", str(config.project_dir / "docker-compose.yml")]
    if needs_build:
        print("Preparando la imagen del traductor; solo se construye al instalar o cambiar dependencias...", flush=True)
        run_command([*compose, "build", "translator"], cwd=config.project_dir, env=config.command_env, timeout=1800,
            progress_message="La preparación de la imagen continúa...")
    old_api = health is not None and not _compatible_health(health)
    source_changed = bool(state.get("source") and state["source"] != source)
    if not needs_build and not configuration_changed and (health is not None or runtime_satisfied) and (source_changed or old_api):
        # Python source is bind mounted. Restarting the existing container loads
        # it while preserving verified packages installed in its writable layer.
        print("Actualizando el servicio del traductor...", flush=True)
        run_command([*compose, "restart", "translator"], cwd=config.project_dir, env=config.command_env, timeout=180)
        _wait_for(lambda: _compatible_health(optional_http_json(config.api_url + "/health")), config.backend_timeout, "Esperando al traductor")
        save_backend_state(config, state["dependencies"] if runtime_satisfied else dependencies, source,
                           runtime_dependencies=dependencies if runtime_satisfied else None)
        return
    up = [*compose, "up", "-d", "--no-build"]
    if old_api or source_changed or configuration_changed:
        up.append("--force-recreate")
    up.append("translator")
    print("Iniciando el servicio del traductor...", flush=True)
    run_command(up, cwd=config.project_dir, env=config.command_env, timeout=180)
    _wait_for(lambda: _compatible_health(optional_http_json(config.api_url + "/health")), config.backend_timeout, "Esperando al traductor")
    save_backend_state(config, dependencies, source)


def prepare_browser_runtime(config: Config) -> Path:
    """Install host dependencies/browser once and return the reusable interpreter."""
    environment_dir = config.project_dir / ".venv"
    python = environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Primera preparación: creando el entorno del navegador...", flush=True)
        run_command([sys.executable, "-m", "venv", str(environment_dir)], cwd=config.project_dir, timeout=180,
            progress_message="La preparación del entorno del navegador continúa...")
    requirements = config.project_dir / "requirements-launcher.txt"
    if not requirements.is_file():
        raise LauncherError("Falta requirements-launcher.txt; no se puede preparar Google Lens.")
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    state_path = config.project_dir / ".cache" / "launcher-host-dependencies.json"
    state = _read_json_file(state_path)
    available = run_command([str(python), "-c", "import playwright"], cwd=config.project_dir, timeout=20, check=False).returncode == 0
    if not available or (state.get("requirements") and state["requirements"] != digest):
        print("Primera preparación: instalando Playwright (puede tardar unos minutos)...", flush=True)
        run_command([str(python), "-m", "pip", "install", "-r", str(requirements)], cwd=config.project_dir, timeout=900,
            progress_message="La instalación de Playwright continúa...")
    browser_check = "from pathlib import Path; from playwright.sync_api import sync_playwright; import sys; p=sync_playwright().start(); ready=Path(p.chromium.executable_path).is_file(); p.stop(); sys.exit(0 if ready else 1)"
    browser_available = run_command([str(python), "-c", browser_check], cwd=config.project_dir, timeout=30, check=False).returncode == 0
    if not browser_available:
        chrome_paths = [base / "Google/Chrome/Application/chrome.exe" for base in _program_files()]
        chrome_paths.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe")
        configured_chrome = config.env.get("GOOGLE_CHROME_PATH")
        chrome = str(Path(configured_chrome)) if configured_chrome and Path(configured_chrome).is_file() else None
        if not configured_chrome:
            chrome = _known_program(chrome_paths, "chrome")
        if chrome:
            LOG.info("Chrome está disponible; no hace falta descargar otro navegador: %s", chrome)
        else:
            print("Primera preparación: descargando Chromium para Google Lens...", flush=True)
            try:
                run_command([str(python), "-m", "playwright", "install", "chromium"], cwd=config.project_dir, timeout=900,
                    progress_message="La descarga de Chromium continúa...")
            except LauncherError:
                LOG.exception("No se pudo descargar Chromium; se intentará Chrome si está disponible.")
                print("No se pudo descargar Chromium. Se intentará Chrome; si tampoco está disponible, se conservará CLEAN.", flush=True)
    _write_json_file(state_path, {"requirements": digest})
    return python.resolve()


def host_post_dir(config: Config, result: dict[str, Any]) -> Path:
    post_id = str(result.get("post_id") or "")
    if not post_id:
        folder = str(result.get("output_dir") or result.get("output_path") or "").replace("\\", "/").rstrip("/")
        post_id = folder.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[0-9]{1,30}", post_id):
        raise LauncherError("El servicio no devolvió una carpeta válida para este post.")
    return config.output_dir / post_id


def _clean_path(post_dir: Path, slide: dict[str, Any]) -> Path:
    relative = PurePosixPath(str(slide.get("clean") or "").replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or ":" in str(relative):
        raise LauncherError("El servicio devolvió una ruta CLEAN no válida.")
    path = (post_dir / Path(*relative.parts)).resolve()
    try:
        path.relative_to(post_dir.resolve())
    except ValueError as exc:
        raise LauncherError("La imagen CLEAN está fuera de la carpeta de resultados.") from exc
    if not path.is_file():
        raise LauncherError(f"No se ha encontrado la imagen CLEAN: {path}")
    return path


def _google_lens_class() -> type:
    module = ".reverse_search.google_lens" if __package__ else "reverse_search.google_lens"
    return importlib.import_module(module, package=__package__ or None).GoogleLensProvider


def _reverse_provider_class(name: str) -> type:
    """Register optional providers here; configuration never imports arbitrary code."""
    factories = {'google_lens': _google_lens_class}
    try:
        return factories[name]()
    except KeyError as exc:
        raise LauncherError(f'Proveedor de búsqueda visual no disponible: {name}.') from exc


def _reverse_payload(result: dict[str, Any], position: int, elapsed: float) -> dict[str, Any]:
    """Keep browser errors/candidates within the backend's request size limits."""
    allowed = {"CANDIDATES", "NOT_FOUND", "CAPTCHA", "UNAVAILABLE", "ERROR", "DISABLED", "TIMEOUT"}
    status = str(result.get("status", "ERROR"))
    error = str(result.get("error") or "")[:2000]
    if status not in allowed:
        status, error = "ERROR", "El proveedor devolvió un estado de búsqueda no válido."
    candidates = []
    raw_candidates = result.get("candidates") or []
    for raw in raw_candidates[:30] if isinstance(raw_candidates, list) else []:
        if not isinstance(raw, dict):
            continue
        url = raw.get("image_url")
        if not isinstance(url, str) or not 0 < len(url) <= 4096:
            continue
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
        except ValueError:
            continue
        source = str(raw.get("source_url") or "")
        candidate = {"image_url": url, "source_url": source if len(source) <= 4096 else "",
            "source_domain": str(raw.get("source_domain") or "")[:300]}
        alternatives = raw.get("alternate_urls") or []
        candidate["alternate_urls"] = [value for value in alternatives[:12]
            if isinstance(value, str) and 0 < len(value) <= 4096] if isinstance(alternatives, list) else []
        for dimension in ("width", "height"):
            value = raw.get(dimension)
            if isinstance(value, int) and 0 < value <= 100_000:
                candidate[dimension] = value
        candidate["thumbnail"] = bool(raw.get("thumbnail", False))
        bridge = raw.get("browser_download")
        if isinstance(bridge, dict):
            # The backend validates status, content type and actual image bytes.
            candidate["browser_download"] = bridge
        candidates.append(candidate)
    if status == "CANDIDATES" and not candidates:
        status = "NOT_FOUND"
    payload = {"position": position, "provider": str(result.get("provider") or "google_lens"),
        "status": status, "error": error, "candidates": candidates,
        "elapsed_seconds": round(min(3600, max(0, elapsed)), 3),
        "query": result.get("query", {}), "cache_hit": bool(result.get("cache_hit", False)),
        "timings": result.get("timings", {}), "diagnostics": result.get('diagnostics', {})}
    for name, limit in (("fresh_search_status", 100), ("fresh_search_error", 2000),
                        ("cache_fallback_from", 100), ("discovery_query_file", 4096),
                        ("discovery_query_sha256", 128)):
        if result.get(name) is not None:
            payload[name] = str(result[name])[:limit]
    return payload


def run_reverse_search(
    config: Config,
    request: dict[str, Any],
    *,
    force: bool = False,
    headless: bool = False,
    provider_factory: Callable[..., Any] | None = None,
    disabled: bool = False,
    session: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    slides = request.get("slides", [])
    if not isinstance(slides, list):
        raise LauncherError("Solicitud de búsqueda visual no válida.")
    provider = session.get('provider') if session is not None else None
    setup_error = None
    provider_name = config.env.get('REVERSE_SEARCH_PROVIDER', 'google_lens').strip().lower() or 'google_lens'
    remaining = min((float(s.get('reverse_query', {}).get('budget_seconds_remaining',
                     _number(config.env, 'REVERSE_TOTAL_BUDGET_SECONDS', 300))) for s in slides), default=300)
    deadline = time.monotonic() + max(0, remaining)
    if session is not None:
        # A round uses the remaining budget supplied by the backend. Reusing the
        # browser must not carry an expired deadline into a later round/post.
        session['deadline'] = deadline
    if not disabled and provider is None:
        try:
            factory = provider_factory or _reverse_provider_class(provider_name)
            provider = factory(
                profile_dir=config.project_dir / ".cache" / "browser" / "google_lens",
                headless=headless,
                timeout_seconds=_number(config.env, "LENS_TIMEOUT_SECONDS", 45),
                interactive_timeout=_number(config.env,
                    "LENS_INTERACTIVE_TIMEOUT" if "LENS_INTERACTIVE_TIMEOUT" in config.env else "LENS_INTERACTIVE_TIMEOUT_SECONDS",
                    120, allow_zero=True),
                max_candidates=int(_number(config.env, "LENS_MAX_CANDIDATES", 12)),
                channel="chrome",
                chrome_path=config.env.get("GOOGLE_CHROME_PATH") or None,
            )
            if session is not None:
                session['provider'] = provider
        except Exception as exc:
            LOG.exception("No se pudo iniciar el proveedor de búsqueda visual %s.", provider_name)
            setup_error = str(exc)
            print("La búsqueda visual no está disponible; se registrarán las imágenes web que falten.", flush=True)
    if provider is not None and hasattr(provider, 'set_deadline'):
        provider.set_deadline(deadline)
    results: list[dict[str, Any]] = []
    try:
        for index, slide in enumerate(slides, 1):
            position = slide.get("position", index)
            started = time.monotonic()
            result: dict[str, Any]
            if disabled:
                result = {"provider": "google_lens", "status": "DISABLED", "candidates": []}
            elif setup_error:
                result = {"provider": provider_name, "status": "UNAVAILABLE", "error": setup_error, "candidates": []}
            elif time.monotonic() >= deadline:
                result = {'provider': provider_name, 'status': 'TIMEOUT', 'candidates': [],
                          'error': 'Se agotó el tiempo de esta ronda de búsqueda visual.'}
            else:
                print(f"       Buscando diapositiva {position} ({index}/{len(slides)})...", flush=True)
                try:
                    query = slide.get("reverse_query") or {"kind": "clean_full", "attempt_index": 1}
                    if query.get("skipped"):
                        result = {"status": "NOT_FOUND", "candidates": [], "query": query}
                    elif query.get("kind") in {"text_assisted", "google_images"}:
                        search_text = getattr(provider, "search_text", None)
                        result = search_text(str(query.get("text_query", "")), force=force) if search_text else {
                            "status": "UNAVAILABLE", "candidates": [], "error": "El proveedor no admite búsqueda textual."}
                    else:
                        relative = query.get("query_file") or slide.get("original" if query.get("kind") == "original_full" else "clean")
                        image_path = _clean_path(host_post_dir(config, request), {"clean": relative})
                        result = provider.search(image_path, force=force)
                    if not isinstance(result, dict):
                        raise LauncherError("Google Lens devolvió un resultado no válido.")
                    prepare_downloads = getattr(provider, "prepare_downloads", None)
                    if callable(prepare_downloads) and result.get("candidates"):
                        prepared = prepare_downloads(result)
                        if isinstance(prepared, dict):
                            result = prepared
                    result = {**result, "query": query}
                except Exception as exc:
                    LOG.exception("Búsqueda visual fallida en diapositiva %s.", position)
                    result = {"provider": "google_lens", "status": "ERROR", "error": str(exc), "candidates": []}
                if result.get("status") in {"ERROR", "UNAVAILABLE", "CAPTCHA", "NOT_FOUND"}:
                    print(f"       Diapositiva {position}: {result['status']}; se intentarán los siguientes métodos disponibles.", flush=True)
                elif result.get("candidates"):
                    print(f"       {index}/{len(slides)} encontrada: {len(result['candidates'])} candidato(s).", flush=True)
            results.append(_reverse_payload(result, position, time.monotonic() - started))
    finally:
        if provider is not None and session is None:
            try:
                provider.close()
            except Exception:
                LOG.debug("No se pudo cerrar el navegador.", exc_info=True)
    return results


_STAGE_LABELS = {
    "download": "[1/6] Descargando imágenes",
    "ocr": "[2/6] Detectando texto",
    "clean": "[3/6] Generando CLEAN",
    "cleaning": "[3/6] Generando CLEAN",
    "reverse_search": "[4/6] Buscando imágenes originales",
    "lens": "[4/6] Buscando imágenes originales",
    "translation": "[5/6] Traduciendo",
    "translate": "[5/6] Traduciendo",
    "output": "[6/6] Guardando",
    "save": "[6/6] Guardando",
    "done": "[6/6] Resultados guardados",
}


def _progress_line(job: dict[str, Any]) -> str:
    progress = job.get("progress") or {}
    stage = str(progress.get("stage") or "")
    label = _STAGE_LABELS.get(stage, str(progress.get("message") or "Procesando"))
    current, total = progress.get("current"), progress.get("total")
    if current is not None and total:
        label += f" ({current}/{total})"
    message = str(progress.get("message") or "")
    if message and stage in _STAGE_LABELS:
        label += f": {message}"
    return label


def _process_job(
    config: Config,
    url: str,
    *,
    force: bool = False,
    reverse_search: bool = True,
    headless: bool = False,
    reverse_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    created = http_json(config.api_url + "/jobs", method="POST", body={"url": url, "force": force, "reverse_search": reverse_search})
    job_id = created.get("job_id") or created.get("id")
    if not job_id or not re.fullmatch(r"[A-Za-z0-9_-]+", str(job_id)):
        raise LauncherError("El servicio no devolvió un identificador de trabajo válido.")
    endpoint = config.api_url + "/jobs/" + str(job_id)
    LOG.info("Trabajo %s iniciado (force=%s, reverse_search=%s).", job_id, force, reverse_search)
    last_progress = None
    reverse_submitted = set()
    while time.monotonic() - started < config.job_timeout:
        job = http_json(endpoint)
        state = str(job.get("status") or job.get("state") or "").upper()
        progress = _progress_line(job)
        if progress != last_progress:
            print(progress, flush=True)
            last_progress = progress
        if state == "COMPLETED":
            result = job.get("result")
            if not isinstance(result, dict):
                raise LauncherError("El trabajo terminó sin un resultado válido.")
            result.setdefault("launcher_elapsed_seconds", round(time.monotonic() - started, 3))
            return result
        if state in {"ERROR", "FAILED"}:
            failed_result = job.get("result") or {}
            LOG.error("Falló el trabajo %s: %r", job_id, failed_result)
            error = job.get("error") or (failed_result.get("error") if isinstance(failed_result, dict) else None) or "El trabajo no se pudo completar."
            if isinstance(error, dict):
                error = error.get("message") or error.get("detail") or str(error)
            raise LauncherError(str(error))
        if state == "WAITING_REVERSE_SEARCH":
            request = job.get("reverse_search_request")
            if not isinstance(request, dict):
                raise LauncherError("El servicio pidió una búsqueda visual sin especificar las imágenes.")
            round_id = request.get("round_id") or "legacy-first-round"
            if round_id in reverse_submitted:
                time.sleep(config.poll_interval)
                continue
            results = run_reverse_search(config, request, force=force, headless=headless, disabled=not reverse_search, session=reverse_session)
            try:
                payload = {"slides": results}
                if request.get("round_id"):
                    payload['round_id'] = request['round_id']
                http_json(endpoint + "/reverse-search", method="POST", body=payload, timeout=30)
            except LauncherError as exc:
                if exc.http_status != 409:
                    raise
                latest = http_json(endpoint)
                latest_state = str(latest.get("status") or latest.get("state") or "").upper()
                if latest_state not in {"RUNNING", "COMPLETED", "ERROR", "FAILED"}:
                    raise
                LOG.warning("Lens entregó candidatos fuera de plazo; el trabajo %s ya está en %s.", job_id, latest_state)
                print("La búsqueda visual llegó fuera de plazo. El procesamiento continúa con las imágenes disponibles.", flush=True)
            reverse_submitted.add(round_id)
        time.sleep(config.poll_interval)
    raise LauncherError(f"El trabajo superó el tiempo de espera ({config.job_timeout:g} s). Consulta los logs del traductor.")


def process_job(config: Config, url: str, *, force: bool = False, reverse_search: bool = True,
                headless: bool = False, reverse_session: dict[str, Any] | None = None) -> dict[str, Any]:
    session: dict[str, Any] = reverse_session if reverse_session is not None else {}
    session.pop('deadline', None)
    provider = session.get('provider')
    reset_for_post = getattr(provider, 'reset_for_post', None)
    if callable(reset_for_post):
        reset_for_post()
    try:
        return _process_job(config, url, force=force, reverse_search=reverse_search,
                            headless=headless, reverse_session=session)
    finally:
        if reverse_session is None:
            _close_reverse_session(session)


def _close_reverse_session(session: dict[str, Any]) -> None:
    provider = session.pop('provider', None)
    if provider is not None:
        try:
            provider.close()
        except Exception:
            LOG.debug('No se pudo cerrar el navegador.', exc_info=True)


def report_result(config: Config, result: dict[str, Any], *, no_open: bool = False) -> None:
    status = str(result.get("status") or "").upper()
    if status.startswith("SKIPPED") or status in {"UNSUPPORTED", "UNSUPPORTED_VIDEO"}:
        print(result.get("error") or "Este post no contiene un carrusel de fotos compatible. No se han extraído fotogramas.", flush=True)
        return
    output = host_post_dir(config, result)
    duration = result.get("launcher_elapsed_seconds", result.get("duration_seconds"))
    print("\nINCOMPLETE." if status == "INCOMPLETE" else "\nCompletado.", flush=True)
    if status == "INCOMPLETE":
        validation = result.get("validation") or {}
        missing = validation.get("missing") if isinstance(validation, dict) else None
        if isinstance(missing, list):
            for detail in missing:
                print(f"Falta: {detail}", flush=True)
        elif missing:
            print(f"Falta: {missing}", flush=True)
        else:
            print(result.get("incomplete_reason") or result.get("error") or "Consulta metadata/manifest.json para ver los archivos que faltan.", flush=True)
    if result.get('review_required'):
        print('Hay resultados que necesitan revisión. Consulta el manifest y las traducciones.', flush=True)
    if result.get('translation_fallbacks'):
        print(f"Se conservó texto en inglés en {int(result['translation_fallbacks'])} diapositiva(s) al fallar todos los traductores.", flush=True)
    if result.get('reverse_pending'):
        print(f"Quedan {int(result['reverse_pending'])} fotografía(s) fuente pendientes; CLEAN sigue disponible.", flush=True)
    if result.get("cached"):
        print("Se han reutilizado los resultados guardados.")
    timings = result.get("timings") or {}
    if isinstance(timings, dict):
        names = {"download": "descarga", "ocr": "OCR", "clean": "limpieza", "cleaning": "limpieza", "reverse_search": "búsqueda visual", "translation": "traducción", "total": "total"}
        values = [f"{names.get(key, key)}: {value:.1f} s" for key, value in timings.items() if isinstance(value, (int, float))]
        if values:
            print("Tiempos: " + "; ".join(values))
    print(f"Resultados: {output}", flush=True)
    if isinstance(duration, (int, float)):
        print(f"Tiempo: {float(duration):.1f} s", flush=True)
    LOG.info("Trabajo terminado: %s", output)
    if not no_open and os.name == "nt" and output.is_dir():
        try:
            os.startfile(str(output))
        except OSError:
            LOG.exception("No se pudo abrir la carpeta de resultados en Explorer.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traducir carruseles de TikTok y recuperar sus fotografías originales.")
    parser.add_argument("--url", help="Enlace de TikTok; si falta, se pide en la consola.")
    parser.add_argument("--loop", action="store_true", help="Pedir otro enlace después de procesar --url.")
    parser.add_argument("--force", action="store_true", help="Reprocesar incluso si el post está guardado.")
    parser.add_argument("--no-open", action="store_true", help="No abrir la carpeta final en Explorer.")
    parser.add_argument("--headless", action="store_true", help="Ocultar el navegador de Google Lens.")
    parser.add_argument("--no-reverse-search", action="store_true", help="Guardar CLEAN sin ejecutar Google Lens.")
    parser.add_argument("--check-only", action="store_true", help="Preparar y comprobar Docker, Ollama y el servicio sin procesar ningún post.")
    return parser.parse_args(argv)


def _prompt_url(*, following: bool = False) -> str | None:
    prompt = "Introduce otro enlace TikTok\n(o escribe SALIR):\n> " if following else "Introduce enlace TikTok:\n> "
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            return None
        if value.lower() in {"salir", "exit", "q"}:
            return None
        if not value:
            continue
        try:
            return validate_tiktok_url(value)
        except LauncherError as exc:
            print(f"Enlace no válido: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    # Windows also uses UTF-8 when output is redirected to a log or a pipe.
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    log_path = setup_logging(root)
    reverse_session: dict[str, Any] = {}
    try:
        if initialize_env(root):
            print("Configuración local creada desde .env.example.", flush=True)
        config = Config.from_project(root)
        print("=================================\nTikTok Image Tool\n=================================", flush=True)
        interactive = args.url is None or args.loop
        url = None
        if not args.check_only:
            url = validate_tiktok_url(args.url) if args.url else _prompt_url()
            if url is None:
                return 0
        if not args.check_only and not args.no_reverse_search and os.environ.get("TIKTOK_LAUNCHER_BOOTSTRAPPED") != "1":
            try:
                interpreter = prepare_browser_runtime(config)
            except Exception as exc:
                LOG.exception("La preparación del navegador falló; se mantendrá CLEAN como alternativa.")
                print(f"No se pudo preparar Google Lens: {exc}\nEl resto del procesamiento continuará.", flush=True)
            else:
                if interpreter != Path(sys.executable).resolve():
                    original_args = list(sys.argv[1:] if argv is None else argv)
                    if args.url is None:
                        original_args.extend(["--url", url, "--loop"])
                    child_env = config.command_env
                    child_env["TIKTOK_LAUNCHER_BOOTSTRAPPED"] = "1"
                    return subprocess.run([str(interpreter), str(Path(__file__).resolve()), *original_args], cwd=root, env=child_env).returncode
        print("Comprobando Docker y Ollama...", flush=True)
        docker = ensure_docker(config)
        ensure_ollama(config)
        ensure_backend(config, docker)
        if args.check_only:
            models = config.model
            if config.translation_model and config.translation_model != config.model:
                models += f"; traducción: {config.translation_model}"
            print(f"Todo preparado. Modelo: {models}. Resultados: {config.output_dir}", flush=True)
            return 0
        while url is not None:
            print("\nProcesando...", flush=True)
            try:
                result = process_job(config, url, force=args.force, reverse_search=not args.no_reverse_search,
                                     headless=args.headless or config.headless, reverse_session=reverse_session)
                report_result(config, result, no_open=args.no_open)
            except LauncherError as exc:
                if not interactive:
                    raise
                LOG.exception("El TikTok no pudo completarse; el launcher sigue abierto.")
                print(f"\nError: {exc}\nDetalle técnico: {log_path}", flush=True)
            if not interactive:
                return 1 if str(result.get("status", "")).upper() == "INCOMPLETE" else 0
            print("\n---------------------------------\n", flush=True)
            url = _prompt_url(following=True)
        return 0
    except KeyboardInterrupt:
        LOG.info("Launcher interrumpido por el usuario.")
        print("\nLauncher interrumpido. El servicio local sigue disponible.", flush=True)
        return 130
    except (LauncherError, EOFError) as exc:
        LOG.exception("Error del launcher.")
        print(f"\nError: {exc}\nDetalle técnico: {log_path}", flush=True)
        return 1
    except Exception as exc:
        LOG.exception("Error inesperado del launcher.")
        print(f"\nError inesperado: {exc}\nDetalle técnico: {log_path}", flush=True)
        return 1
    finally:
        _close_reverse_session(reverse_session)


if __name__ == "__main__":
    raise SystemExit(main())
