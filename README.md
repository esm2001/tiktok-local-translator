# TikTok Local Translator

## Overview

Aplicación local para Windows que descarga carruseles de fotografías de TikTok, detecta su texto, crea imágenes CLEAN y traduce al castellano. Google Lens busca la fotografía fuente; se descarga el mejor candidato disponible sin imponer un umbral de similitud. SOURCE conserva los bytes web y su formato, sin limpieza ni traducciones superpuestas.

La traducción se entrega exclusivamente en `traducciones.txt`. Ollama recibe texto. La aplicación no dibuja traducciones sobre imágenes, no genera imágenes `*_ES.jpg` y no extrae fotogramas de vídeos.

El estado de las pruebas y las mediciones pertenece a [VALIDATION_REPORT.md](VALIDATION_REPORT.md) y [BENCHMARK.md](BENCHMARK.md). Consulta esos informes para distinguir resultados comprobados, casos pendientes y limitaciones. [VERIFICATION.md](VERIFICATION.md) conserva la verificación histórica.

Actualización del 12 de octubre de 2026: traducción directa del OCR mediante Marian/OPUS EN→ES, después Google Translate web como respaldo opcional, después Ollama y, si todos fallan, inglés con aviso explícito. El launcher pide otro enlace al terminar; `salir`, `exit` o `q` cierran la consola. Docker, OCR, modelo local y navegador se reutilizan. Los informes anteriores y la sección histórica de traducción describen las mediciones previas a este cambio.

Modelo local: [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es), con conversión int8 publicada por [playto-mt](https://huggingface.co/playto-mt/opus-mt-en-es-ct2). Se conservan su revisión, URL de descarga y hashes en `translator/models/opus-mt-en-es-ct2/download-provenance.json`.

## Architecture

```text
Windows: launcher + Chrome/Chromium persistente
    │                              │
    │ API local                    └─ Google Lens: candidatos
    ▼
Docker: FastAPI
    gallery-dl → ORIGINAL → PaddleOCR → máscara → CLEAN
                      │                    │
                      │                    └─ ranking → descarga web RAW → SOURCE
                      └─ texto OCR → Marian → Google/Ollama si falla
    manifest.json + traducciones.txt
```

El navegador permanece en Windows para permitir consentimientos y resolución manual de CAPTCHA. Docker conserva una instancia OCR en memoria y procesa un carrusel cada vez. Ollama funciona en Windows, con inferencia por CPU y sin recibir imágenes.

El backend HTTP se publica en `127.0.0.1`. El código Python se monta en el contenedor; cambiar código puede reiniciar el servicio sin reconstruir su imagen. Solo se construye cuando falta la imagen o cambian sus dependencias. Los volúmenes Docker conservan modelos OCR.

## Requirements

- Windows 10/11 de 64 bits.
- [Python 3.11 o posterior](https://www.python.org/downloads/windows/), disponible como `python` en PowerShell.
- [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) con contenedores Linux y Docker Compose.
- [Ollama para Windows](https://docs.ollama.com/windows) y los modelos configurados.
- Chrome, o Chromium instalado por Playwright. Playwright se instala en el entorno Python del proyecto.
- Git para clonar el repositorio. GitHub CLI solo es necesario para administrar o publicar el repositorio.

No se necesita GPU dedicada ni una clave de API. Docker, modelos OCR y modelos Ollama ocupan varios GB. La RAM necesaria y el tiempo dependen del modelo y del contenido; los resultados del equipo de prueba están en el benchmark.

## Installation

Clona el repositorio privado en una carpeta de tu elección, por ejemplo `C:\Tiktok\`, y ejecuta:

```powershell
git clone <URL_DEL_REPOSITORIO_PRIVADO>
cd tiktok-local-translator
.\setup.ps1
```

`setup.ps1` comprueba Python, prepara `.venv`, instala las dependencias pequeñas del navegador con aviso, crea `.env` desde `.env.example` si falta y prepara directorios. Comprueba Docker, Ollama, los modelos y Chrome/Chromium. Conserva una configuración existente.

En `setup.ps1`, las descargas grandes requieren opciones explícitas. En una instalación nueva suele hacer falta:

```powershell
.\setup.ps1 -PullModels -BuildBackend
```

Ese comando puede descargar varios GB de modelos Ollama, capas Docker y modelos OCR. Si Chrome está instalado, se reutiliza. Para descargar Chromium cuando no tienes Chrome:

```powershell
.\setup.ps1 -InstallBrowser
```

Opciones adicionales:

```powershell
.\setup.ps1 -CheckOnly       # Comprueba; no instala ni inicia servicios
.\setup.ps1 -Development     # Instala también requirements-dev.txt
```

Si PowerShell bloquea scripts locales, puedes ejecutar el script concreto con:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

Docker Desktop y Ollama se instalan por separado. El setup informa si faltan; no descarga sus instaladores. Las siguientes ejecuciones reutilizan dependencias, imágenes Docker y modelos.

## Configuration

Edita tu `.env` local. No lo añadas a Git. [`.env.example`](.env.example) documenta las variables disponibles.

| Variable | Uso |
| --- | --- |
| `OUTPUT_HOST_DIR` | Resultados en Windows; ruta absoluta o relativa al proyecto. Por defecto `../TikTokTraducidos`. |
| `OLLAMA_HOST_URL` | Ollama desde Windows, normalmente `http://localhost:11434`. |
| `OLLAMA_BASE_URL` | Ollama desde Docker, normalmente `http://host.docker.internal:11434`. |
| `OLLAMA_MODEL` | Modelo local para reconstruir texto OCR. |
| `OLLAMA_TRANSLATION_MODEL` | Modelo de traducción; vacío reutiliza el principal. |
| `OLLAMA_KEEP_ALIVE` | Tiempo que Ollama mantiene el modelo cargado. |
| `OCR_CPU_THREADS` | Hilos CPU utilizados por PaddleOCR. |
| `OCR_ADAPTIVE_FALLBACK` | Activa fallbacks según evidencia de pérdida; por defecto `true`. |
| `OCR_MAX_CROP_RETRIES` | Máximo de recortes OCR por imagen; por defecto `6`, `0` los desactiva. |
| `REVERSE_MAX_QUERIES` | Máximo de variantes de consulta por diapositiva; por defecto `6`. |
| `REVERSE_TOTAL_BUDGET_SECONDS` | Presupuesto por diapositiva y ronda de fallback; por defecto `300`. |
| `REVERSE_NEGATIVE_TTL_SECONDS` | Vigencia de búsquedas negativas completas; por defecto `3600`. |
| `REVERSE_SEARCH_WAIT_SECONDS` | Plazo del navegador/proveedor, sujeto al presupuesto restante de búsqueda. |
| `TRANSLATION_TIMEOUT_SECONDS` | Plazo máximo de una petición textual a Ollama; por defecto `180`. |
| `TRANSLATION_BUDGET_SECONDS` | Presupuesto conjunto de reconstrucción, traducción y reintentos por post; por defecto `360`. |
| `TRANSLATION_RETRY_TTL_SECONDS` | Espera antes de volver a intentar una salida incompleta; por defecto `60`. |
| `TRANSLATOR_PORT` | Puerto local del backend. |
| `LENS_HEADLESS` | Navegador visible u oculto; visible facilita resolver CAPTCHA. |
| `LENS_MAX_CANDIDATES` | Límite de candidatos por consulta. |
| `VISUAL_MATCH_THRESHOLD` | Umbral visual, acompañado de condiciones geométricas y control OCR. |
| `CANDIDATE_MIN_DIMENSION` | Resolución mínima del candidato. |

La configuración inicial mantiene `gemma3:4b`; compara los modelos con el mismo dataset antes de cambiarla. `GOOGLE_CHROME_PATH` permite indicar otro Chrome. Los plazos de Docker, backend, Ollama, Lens y trabajos también son configurables.

Las cookies opcionales de gallery-dl pertenecen a `secrets/cookies.txt`. Esa carpeta se monta como solo lectura en Docker. No compartas cookies, perfiles ni datos de sesión.

## Running

```powershell
.\run.ps1
.\run.ps1 --url "https://www.tiktok.com/@cuenta/photo/1234567890123456789"
.\run.ps1 --url "https://vm.tiktok.com/XXXXXXXX/" --force
.\run.ps1 --no-reverse-search
.\run.ps1 --check-only
```

El launcher pide un enlace si no lo proporcionas, comprueba los servicios, muestra progreso y abre la carpeta final. `--no-open` evita abrir Explorer. `--headless` oculta el navegador. También puedes usar `python launcher/main.py` o abrir `launcher/start.cmd`. Si PowerShell bloquea `run.ps1`, usa `powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1`.

Si no encuentra Chrome ni Chromium, el launcher avisa y descarga Chromium antes de abrir el navegador. Para preparar esa descarga por separado, utiliza `setup.ps1 -InstallBrowser`.

`--no-reverse-search` conserva OCR, CLEAN y traducción. `--force` reprocesa el post y evita reutilizar la búsqueda visual. `--check-only` del launcher prepara los servicios; utiliza `setup.ps1 -CheckOnly` cuando solo quieras inspeccionarlos.

## Output structure

```text
TikTokTraducidos/<POST_ID>/
├─ original/
│  ├─ 01.jpg
│  └─ 02.jpg
├─ clean/
│  ├─ 01_CLEAN.jpg
│  └─ 02_CLEAN.jpg
├─ recovered/
│  └─ 01_SOURCE.jpg
├─ traducciones.txt
└─ metadata/
   ├─ manifest.json
   └─ masks/
      ├─ 01_MASK.png
      └─ 02_MASK.png
```

Se espera un SOURCE por diapositiva, con extensión `.jpg`, `.png` o `.webp` según el archivo web. CLEAN permanece separado. `best_image` identifica únicamente SOURCE y queda vacío si todas las búsquedas/descargas fallan. Se validan los tres recuentos y las secciones de `traducciones.txt`; si falta algo, el resultado es `INCOMPLETE` y detalla la diapositiva afectada.

El manifest guarda OCR y polígonos, reconstrucción, traducción, incertidumbre, intentos de búsqueda, candidatos, motivos de rechazo y tiempos. Los errores quedan visibles; una salida incompleta no se presenta como verificada.

## OCR pipeline

El OCR principal combina el detector `PP-OCRv5_mobile_det` y el reconocedor inglés `en_PP-OCRv5_mobile_rec`. Detección y reconocimiento se evalúan por separado: encontrar una región no garantiza haber leído correctamente sus caracteres. La selección y sus medidas se documentan en [BENCHMARK.md](BENCHMARK.md).

El camino rápido procesa la imagen original. Los casos con señales de pérdida o baja confianza activan los fallbacks configurados; los recortes o preprocesados adicionales deben justificar su coste con el dataset. Las máscaras se usan para limpiar texto mediante OpenCV y para excluirlo al comparar candidatos.

Una diapositiva con texto visible y OCR vacío es un fallo de regresión. El ground truth de las imágenes locales pertenece a `tests/regression/expected.json`; los casos ambiguos se identifican explícitamente.

## Historical translation pipeline (previous benchmarks)

La reconstrucción recibe todas las líneas de una diapositiva para conservar contexto, orden, nombres y pies. La traducción recibe texto reconstruido; las llamadas a Ollama se realizan con `num_gpu: 0`.

La política prioriza respetar OCR, corregir errores sustentados por evidencia y conservar significado. Las líneas partidas se reúnen cuando forman una cláusula; listas y títulos conservan sus separaciones. Los términos breves entre comillas se mantienen literalmente. Los fragmentos con OCR débil o ambigüedad declarada se conservan en inglés y se registran en el manifest; las declaraciones del modelo deben corresponder a un fragmento exacto del original.

Los encabezados breves cuyo comparativo no tiene un referente inequívoco también quedan protegidos en inglés. Un marcador ya protegido puede declararse de nuevo sin invalidar el cuerpo español, siempre que se conserve exactamente y en su orden. Marcadores inventados, alterados o ampliados se rechazan.

El esquema JSON permite declarar fragmentos inciertos. Antes de aceptar la respuesta, el código comprueba que cada fragmento exista literalmente en esa misma diapositiva y se conserve en el destino; rechaza fuentes inventadas, omisiones y cambios de marcadores. Estas comprobaciones no certifican por sí solas que el castellano conserve el sentido o sea natural.

Controles léxicos acotados detectan pérdidas de objeto en `self-sabotage`, calcos espaciales de `sleep on` ante decisiones y el uso aislado de «sin entrada» para estímulos cerebrales. Quedan fuera los objetos físicos y contextos técnicos. También se comprueba la relación de tema en `write about`, distinta de escribir ese objeto directamente. El único reintento puede recibir una explicación del sentido en inglés; no se sustituyen frases por traducciones prefijadas. Si persiste el fallo, se conserva la diapositiva en inglés y se registra la causa.

Un encabezado formal de `sleep on` también debe conservar el consejo de aplazar la decisión; reflexionar o soñar por sí solos pierden esa relación. Una comprobación local de concordancia detecta determinantes masculinos junto al sustantivo femenino `jerga`, sin modificar menciones literales protegidas. Estos controles cubren errores observados, no toda la gramática española.

Una comprobación de naturalidad revisa el patrón imperativo `force` con objeto nominal explícito y `to` cuando el destino usa un reflexivo «a que» seguido de un sujeto nominal tercero. Se limita a esa combinación y excluye instrucciones dirigidas a `yourself`, citas y formas causativas aceptables. Señala una formulación forzada; no afirma por sí sola pérdida de significado ni una prohibición general de los reflexivos.

Si se reciben dos reconstrucciones que los controles rechazan, todavía puede evaluarse la traducción desde las líneas OCR literales cuando toda la evidencia sin protección tiene confianza operativa de al menos 0,90. La fuente literal pasa sus propios controles y conserva cifras, negaciones y marcadores. El manifest registra `OCR_LITERAL_AFTER_MODEL_REJECTION`, la reconstrucción rechazada y el resultado de traducción por separado. Errores de servicio, JSON, truncación, contexto o presupuesto no habilitan esta continuación; el reloj y el único reintento de traducción se mantienen.

La confianza de reconstrucción usa evidencia de OCR y alineación, independiente de la confianza declarada por el modelo. La adecuación y naturalidad del castellano quedan `UNVERIFIED`, con confianza numérica `null`; los controles de omisiones, cambios de actor y otros errores medidos no certifican equivalencia semántica universal. Una salida que no supera los controles conserva el inglés y el motivo del fallo.

Los validadores y reintentos están acotados. Las pruebas deben detectar omisiones y traducciones fluidas que cambian el significado. El benchmark compara reconstrucción, traducción, errores y tiempo sobre los mismos textos reales.

## Reverse image search

Google Lens es el proveedor principal. Playwright automatiza su interfaz web y conserva un perfil dedicado en `.cache/browser/google_lens/`, separado del Chrome habitual. Google recibe las imágenes utilizadas en las consultas; Ollama recibe solamente texto.

La estrategia intenta CLEAN completa, ORIGINAL, dos recortes sin texto, Google Images y una búsqueda textual apoyada en OCR. Si Google bloquea la sesión, prueba páginas públicas de Google/Bing y búsqueda textual en Wikimedia. Para en cuanto consigue candidatos descargables. Cada intento, URL de página e imagen, resolución, score, respuesta HTTP y archivo elegido aparecen en el manifest y los logs.

El backend exige HTTP 200, Content-Type `image/*`, tamaño razonable y decodificación válida. Prueba URLs alternativas, imágenes de la página origen y Referer; Playwright aporta cookies/headers cuando hace falta. Descarga concurrentemente y elige el mayor score, penalizando texto y thumbnails. El OCR del candidato solo afecta al ranking: nunca borra sus píxeles. El resultado se copia sin recodificar y registra `source=web_download`, `raw_download=true`, `modified=false` y SHA256.

El presupuesto de búsqueda se propaga al proveedor, a la transferencia y a las comprobaciones entre candidatos. Es cooperativo: una operación OpenCV u OCR que ya se está ejecutando puede terminar después del plazo; no constituye un límite estricto de tiempo de pared.

Un score bajo registra `confidence=LOW` y conserva el archivo. Si ninguna URL resulta descargable tras los fallbacks, el resultado es `INCOMPLETE`; no se coloca CLEAN en recovered. CAPTCHA, indisponibilidad y errores HTTP quedan registrados.

Un CAPTCHA se resuelve manualmente en el navegador visible. No se utilizan APIs de pago. Google Translate web no es una API oficial garantizada: si devuelve un error se utiliza el siguiente traductor.

## Cache

- El pipeline conserva descargas, OCR, CLEAN y traducción compatibles con configuración y hashes.
- Las fuentes validadas se reutilizan si mantienen su integridad y las condiciones de validación.
- Lens conserva candidatos en `.cache/reverse_search/`; errores y CAPTCHA no deben convertirse en resultados negativos permanentes.
- Docker conserva la imagen y los modelos OCR; el launcher no reconstruye Docker por cada TikTok.
- `--force` permite reprocesar y mantiene una copia temporal para restaurar el resultado anterior si falla.

Cachés, perfiles, modelos, medios descargados y resultados están excluidos de Git. No borres evidencia diagnóstica de un caso pendiente antes de investigar su fallo.

## Debugging

```powershell
.\.venv\Scripts\python.exe tools/debug_slide.py <POST_ID> 3
docker compose logs --tail 100 translator
.\setup.ps1 -CheckOnly
```

La herramienta de diapositivas muestra texto OCR, reconstrucción, traducción, intentos y candidatos desde el manifest. Revisa `logs/launcher.log` y `metadata/manifest.json` para localizar la fase concreta del error.

`tools/verify_download.py` verifica gallery-dl en el contenedor. `tools/verify_recovery_public.py` comprueba descarga y validación de una fuente pública conocida en un directorio aislado; esa prueba por sí sola no demuestra que Lens la haya encontrado.

La comprobación completa aislada descarga otra copia de los carruseles y ejecuta el pipeline real con OCR, CLEAN, validación de fuentes y Ollama:

```powershell
.\.venv\Scripts\python.exe tools/validate_pipeline_e2e.py --container tiktok-translator
.\.venv\Scripts\python.exe tools/validate_pipeline_e2e.py --container tiktok-translator --holdout tests/regression/holdout.json
```

Los resultados JSON y logs se guardan en `.cache/regression/e2e/`; los posts de prueba quedan bajo `.cache/regression/e2e/` del volumen de fotografías. Las fuentes conocidas se entregan mediante un replay explícito de descubrimiento y transferencia local: esta comprobación mide el pipeline y su validador, y deja la búsqueda Lens real pendiente. Se comprueba que los archivos de los posts existentes conservan sus hashes. `--download-only --url URL` permite probar un nuevo carrusel sin OCR ni Ollama y sin inventar ground truth.

`--holdout` añade la muestra nueva sin modificar el dataset de comparación inicial. Después de cada post se ejecuta una repetición real con `force=False`, contando cualquier nueva descarga, OCR, búsqueda o llamada Ollama y comprobando hashes y punteros SOURCE. Un caso sin overlay editorial conserva los logos como evidencia y queda fuera de las tasas de reconstrucción y traducción.

Para comprobar un cambio de máscaras o traducción sobre una ejecución anterior, `--revalidate-existing .cache/regression/e2e/<RUN_ID>/posts` copia sus posts a un sandbox nuevo, verifica los hashes y ejecuta `run_pipeline(force=False)` con el código actual. Las descargas, la resolución de URLs y las transferencias de candidatos quedan bloqueadas; las fuentes guardadas se validan nuevamente cuando cambian sus condiciones. La evidencia anterior permanece intacta. El reporte distingue esta revalidación de una ejecución con descargas nuevas y separa el tiempo del pipeline en caché del coste de comprobar hashes.

El diagnóstico de orden JSON prepara dos variantes de traducción sobre una revalidación v14 conservada:

```powershell
.\.venv\Scripts\python.exe tools/benchmark_translation_schema_order.py --source-report .cache/regression/e2e/v14-final.json --dry-run
```

La preparación es local y no llama a Ollama. `--execute` realiza la comparación con el mismo inglés medido y los mismos validadores, capturando las respuestas finales y el orden real de los campos, sin incluir `thinking`. Requiere los informes y medios locales correspondientes; no descarga fixtures ausentes. Compara 14 diapositivas editoriales con presupuesto nuevo de traducción, sin repetir OCR ni reconstrucción. Sus tiempos y controles de cláusulas no sustituyen el end-to-end ni la revisión bilingüe.

## Known limitations

TikTok, Google y los servidores de imágenes pueden cambiar, bloquear peticiones o imponer límites. Una prueba local no garantiza éxito universal. Solo los casos realmente ejecutados cuentan en las tasas del benchmark.

La aceptación actual es parcial: el descubrimiento Lens requiere resolver un CAPTCHA y aún falta cobertura de traducción castellana en casos legibles. Un fallback inglés seguro no cuenta como una traducción española correcta. [VALIDATION_REPORT.md](VALIDATION_REPORT.md) identifica las pruebas y casos pendientes antes de publicar el proyecto.

Texto ambiguo, fotografías con pocas características y diferencias grandes de composición pueden requerir revisión. La traducción automática puede fallar semánticamente aunque sea fluida. CLEAN elimina texto detectado; no garantiza reconstruir perfectamente una parte del fondo oculta.

Un perfil persistente solo debe abrirse desde una instancia del navegador. Si Lens no está disponible, se conserva CLEAN y se registra el motivo.

## Development

```powershell
.\setup.ps1 -Development
.\.venv\Scripts\python.exe -m pytest tests -q
```

Para comparar modelos OCR pequeños sobre el dataset local, sin reconstruir Docker ni cambiar la configuración de producción:

```powershell
.\.venv\Scripts\python.exe tools/benchmark_ocr_candidates.py --container tiktok-translator --allow-downloads
```

`--allow-downloads` autoriza las descargas de esos modelos OCR pequeños. El benchmark ejecuta cada candidato por CPU en un proceso separado, registra detección, CER, tiempo y RSS, y guarda resultados en `.cache/regression/`. Usa las fotografías del volumen existente; no las envía a un servicio OCR externo. Puedes indicar `--media-root`, `--container-media-root` y `--case` para otras ubicaciones o una selección de diapositivas.

Docker utiliza `opencv-contrib-python==4.10.0.84`, requerido por PaddleX. No se instala otro wheel de OpenCV en el mismo entorno porque comparten `cv2`. Las pruebas de desarrollo sin PaddleX usan el wheel headless de la misma versión. El build ejecuta `pip check` y verifica la distribución única y la versión importada.

```powershell
.\.venv\Scripts\python.exe tools/verify_runtime_dependencies.py --container tiktok-translator
```

```text
launcher/                 Orquestación Windows y proveedores de búsqueda
translator/src/           OCR, limpieza, matcher, recuperación y traducción
tests/                    Pruebas funcionales y de regresión
tests/regression/         Ground truth textual y referencias locales
tools/                    Diagnóstico, verificaciones y benchmarks
docker-compose.yml        Servicio translator; n8n opcional
setup.ps1 / run.ps1        Preparación y entrada portable
```

Guarda fotografías de regresión fuera del repositorio o en `fixtures/local/`, excluida de Git. Se versionan expectativas textuales y herramientas, sin incluir medios ajenos, cookies, perfiles, modelos, `.env` o logs.

Añade una regresión para cada fallo corregido. Mide detección, reconocimiento/reconstrucción, traducción, recuperación, falsos positivos, falsos negativos y tiempos por fase. Actualiza [BENCHMARK.md](BENCHMARK.md) y [VALIDATION_REPORT.md](VALIDATION_REPORT.md) con resultados reproducibles y sus límites.

El repositorio debe permanecer PRIVATE. Antes de publicarlo, revisa `git status`, `git diff` y `git ls-files`, y comprueba que todos los criterios de aceptación requeridos han pasado.
