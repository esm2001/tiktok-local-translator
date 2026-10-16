# Verificación local

Este informe es histórico, del 2 de octubre de 2026. Para la fase de robustez del 3 de octubre, consulta [VALIDATION_REPORT.md](VALIDATION_REPORT.md) y [BENCHMARK.md](BENCHMARK.md); los resultados antiguos no representan la aceptación de los cambios actuales.

Fecha: 2 de octubre de 2026. Equipo Windows, Ollama por CPU e Intel UHD. Se probaron las siete fotografías del post `7659483315203804449` aportadas en la carpeta hermana `TikTokTraducidos`.

## Comprobaciones realizadas

- Suite final: **173 pruebas correctas** en 28,21 s. Una advertencia de deprecación de Starlette/httpx en el entorno de tests; no hubo fallos.
- Inspección del proyecto anterior, sus dependencias instaladas, Docker, mounts, configuración y resultados históricos antes de modificarlo.
- Resolución y descarga real con gallery-dl: siete fotografías, 1,05 s para resolver y 2,93 s para descargar. Descarga temporal; no extracción de fotogramas.
- OCR real con PP-OCRv5 mobile y limpieza TELEA de las siete fotografías. Los modelos se conservan en volúmenes Docker y hay una única instancia OCR por servicio.
- Los recortes nativos con margen de diez píxeles corrigieron `Which one are your` → `Which one are you?` y `differentlyo` → `differently.`: confianza 0,899 → 0,953 y 0,922 → 0,952. Coste total 0,95 s para seis recortes; las otras 31 lecturas y todos los polígonos permanecieron iguales. No se sustituyó el OCR.
- Validación visual con las fotografías reales y transformaciones controladas: cinco fondos coincidentes aceptados y trece casos negativos rechazados. Incluye crop, escala, JPEG, variación de color, imágenes distintas, texto conservado y CLEAN.
- Los siete JPEG antiguos con traducción superpuesta se rechazaron como fuentes. Tres superaban la comparación de fondo; el segundo control con PaddleOCR detectó el texto y evitó aceptarlos.
- Recuperación real completa de la ilustración 7: un candidato público localizado por su título en Gutenberg pasó descarga HTTP, ORB/RANSAC, comparación del fondo, PaddleOCR y escritura de SOURCE en un directorio temporal. Resultado `MATCH`, score 0,9018, JPEG de 500×674 y 138.372 bytes. Sus bytes coinciden con la descarga y los 23 archivos del post permanecieron idénticos durante la prueba; 14 comprobaciones correctas en 6,63 s. Esta prueba valida el backend de recuperación y no usa Lens. [Libro fuente](https://www.gutenberg.org/files/60859/60859-h/60859-h.htm), [imagen pública](https://www.gutenberg.org/files/60859/60859-h/images/p3.jpg).
- API de trabajos, progreso, espera del navegador, exclusión de trabajos simultáneos, errores y fallback; launcher y parser Lens; integridad de caché, reparación y conservación del resultado anterior ante fallos de `--force`.
- Entrega tardía de Lens: el backend continúa con CLEAN al agotar su espera; un HTTP 409 al enviar tarde los candidatos permite que el launcher siga el resultado del trabajo, conservando los errores auténticos.
- Caché positiva de SOURCE: las pruebas verifican que un reintento busca sólo diapositivas pendientes, conserva fuentes con hashes y validación intactos y vuelve a recuperar archivos alterados o ilegibles. Un cambio de modelo de traducción conserva una fuente compatible; `--force` la revalida.
- Arranque real del launcher, backend saludable y reutilización de contenedor. Una repetición del post completado tardó 4,1 s desde el launcher y 0,3 s en el backend, sin descargas, OCR, llamadas Ollama ni build.
- La carpeta `translated/` histórica se trasladó a la caché de archivo fuera del post exportado. La aplicación no contiene funciones para dibujar texto ni crea JPEG traducidos.

Las pruebas automáticas usan mocks para servicios externos y archivos locales reales para las comprobaciones visuales. Se indican por separado las pruebas reales con Docker, gallery-dl, PaddleOCR y Ollama; pasar los mocks no demuestra una búsqueda real en Google.

El arranque de Docker Desktop y Ollama desde estado detenido se comprueba con mocks y plazos controlados; la prueba real reutilizó los servicios activos. No se detuvo Docker Desktop, donde también estaba funcionando n8n, para provocar esa situación. Las ejecuciones de QA usaron `--no-open`: la apertura de Explorer está implementada, pero no se abrió una ventana durante esas pruebas.

## Medición inicial

Antes de cambiar el servicio se midió su procesamiento de las siete imágenes existentes:

| Fase | Segundos |
| --- | ---: |
| OCR | 18,95 |
| Limpieza | 7,34 |
| Traducción con qwen2.5:3b | 108,14 |
| Total | 134,46 |

No incluye descarga ni Lens. La traducción era el principal coste. La prueba del modelo `qwen2.5:1.5b` tardó 46,3 s pero omitió nombres y pies de imagen; se descartó por calidad. Las comprobaciones de integridad de texto evitan devolver algunos resultados truncados o con inglés sin traducir, pero no certifican equivalencia semántica: las traducciones del caso real también se revisan frase por frase.

La comparación de Qwen 3B y Gemma 3 4B con el mismo inglés y prompt favoreció a Gemma para castellano. Una ejecución posterior con Qwen para reconstrucción y Gemma para castellano terminó en 113,44 s, pero la revisión manual encontró «Lo viví / No cambié» en un pie referido a Nietzsche. Se añadió una comprobación genérica de persona con reintento limitado; los tests incluyen citas legítimas en primera persona y palabras acentuadas que no son verbos.

En una llamada real de reconstrucción sobre las 32 líneas editoriales, Gemma conservó las siete diapositivas y corrigió `realty` a `reality`; tardó 17,27 s con el modelo cargado. Qwen había mantenido ese error. Gemma todavía conservó `comfortzone`, por lo que el prompt ahora pide separar palabras unidas por OCR usando un ejemplo ajeno al post (`livingroom`). Se configura Gemma para ambas etapas, siempre con `num_gpu: 0`, evitando dos cargas de modelos por carrusel. No se sustituyeron PaddleOCR, TELEA ni Ollama.

## Exportación real con la configuración final

El trabajo por la API local guardó siete originales, siete CLEAN, siete máscaras y `recovered/07_SOURCE.jpg`, además de texto y manifest. El candidato 7 se entregó expresamente por la API desde la fuente pública encontrada por título: el manifest registra `text_search_public_source`, score 0,9018 y SHA256 verificado. Las otras seis búsquedas se marcaron `DISABLED`; `reverse_search_completed` permanece falso. Este resultado no se atribuye a Lens.

| Fase | Segundos |
| --- | ---: |
| Resolución y descarga | 0,00; fotografías ya descargadas |
| OCR | 5,01; reutilización y recortes de baja confianza |
| Limpieza | 0,14; artefactos intactos reutilizados |
| Recuperación | 2,81; candidato público conocido, sin búsqueda Lens |
| Reconstrucción y traducción | 82,56 |
| Total backend | 91,05 |

Hubo dos llamadas a Gemma, sin reintentos ni cambios de modelo: reconstrucción 26,73 s y castellano 55,77 s; 9,17 s de carga total del modelo. La repetición completa en caché tardó 0,3 s en backend y 4,1 s en la fase de trabajo del launcher, excluyendo la preparación de Docker/Ollama. No hubo build, descarga, OCR ni generación Ollama en esa repetición.

Un reintento parcial real posterior solicitó únicamente las seis diapositivas pendientes. La 7 conservó su SOURCE con `cache_hit=true`, los mismos bytes y `best_image=recovered/07_SOURCE.jpg`; no repitió descarga ni OCR del candidato. También reutilizó el texto con `translation_cache_hit=true`: la fase de traducción fue 0,00 s. El trabajo tardó 8,99 s tras reiniciar el servicio, incluida la carga OCR para los recortes pendientes. El manifest conserva las métricas de la generación original cuando reutiliza el texto; ese indicador permite distinguirlas de llamadas nuevas. Las 25 pruebas de pipeline pasaron de nuevo después de añadir el indicador.

La revisión manual e independiente confirma que las siete traducciones conservan contenidos, nombres, pies y tiempos. El pie 3 mantiene tercera persona y «sobre ello»; el 5 mantiene pasado y objeto singular. Queda una limitación editorial: «Te saboteas cada buena cosa en tu vida» en la 4 es comprensible, pero poco idiomático. En el inglés también persisten `comfortzone` unido y `Until` con mayúscula interna. No se presenta el resultado como una traducción revisada para publicación. Una prueba adicional de prompt omitió nombres y pies sin mejorar esa frase, por lo que se descartó y se conservó la versión íntegra. No se sustituyó ninguna salida por texto escrito manualmente.

Estas mediciones pertenecen a un solo carrusel y mezclan fases reutilizadas con una generación nueva. No son una comparación idéntica con el procesamiento inicial completo ni incluyen la búsqueda real de Google.

También se probó una revisión editorial autónoma del castellano con Gemma: una llamada de 23,36 s, siete validaciones correctas y siete textos idénticos a los anteriores. No mejoró la frase literal de la 4. Se descartó añadir esa etapa porque aumentaba el tiempo sin beneficio de calidad demostrado.

Para la diapositiva 3 se probaron dos enlaces públicos de Wikimedia encontrados mediante búsqueda textual. El primer retrato era otra fotografía y se rechazó por evidencia geométrica insuficiente. El segundo enlace devolvió HTTP 429: no se pudo descargar ni validar. Se conserva CLEAN para esa diapositiva y no se declara una fuente por su título o apariencia probable. Los informes quedan en `.cache/recovery-public-probe/source-03-summary.json`.

## Comprobación externa pendiente

El proveedor Google Lens está implementado con Playwright y perfil visible persistente. Chrome abrió, pero la red del entorno de desarrollo devolvió `ERR_NETWORK_ACCESS_DENIED`. Un intento posterior de probar la subida fuera de ese bloqueo fue rechazado por la revisión automática de permisos: exigía autorización específica para enviar `TikTokTraducidos/7659483315203804449/clean/03_CLEAN.jpg` a Google Lens. Se ha solicitado esa autorización y sigue pendiente.

Por tanto, no se ha demostrado todavía una recuperación real completa desde Lens. No se inventan URLs fuente ni archivos SOURCE para rellenar resultados. Este es el punto pendiente del criterio de aceptación del MVP. Ante cualquier error de búsqueda, el flujo conserva CLEAN y termina el resto del carrusel; los errores transitorios permiten reintentar Lens reutilizando OCR y traducción.

La interfaz de Google puede cambiar y requerir ajustes. La selección es conservadora: fondos con pocas características, grandes diferencias de composición o texto de origen que coincida con las regiones enmascaradas pueden rechazarse aunque sean útiles. El score registrado es diagnóstico, no una probabilidad calibrada.

## Utilidades reproducibles

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
python launcher/main.py --check-only
python launcher/main.py --url "https://vm.tiktok.com/ZGdQG5uEr/" --no-reverse-search
```

`tools/verify_download.py` comprueba gallery-dl en el contenedor; `tools/verify_recovery_existing.py` comprueba el rechazo de los JPEG históricos con OCR real; `tools/verify_recovery_public.py` repite la recuperación completa aislada del candidato público de Gutenberg y admite `--slide-position`, `--image-url` y `--source-url` para otros candidatos. `tools/benchmark_translation.py` compara modelos instalados sobre el OCR del post local sin cambiar su configuración. Los informes detallados se guardan en `.cache/` y los logs del launcher en `logs/`; ambos quedan excluidos del futuro repositorio.

No se ha inicializado Git, creado un repositorio remoto, publicado el proyecto ni empaquetado el EXE.
