# Benchmark de robustez

Actualizado: 4 de octubre de 2026. Estado: **PARTIAL**, aceptación de traducción incompleta y Lens pendiente de resolución manual de CAPTCHA. El informe de aceptación es [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

El último pase completo de producción, v14, conserva las 77 regiones editoriales y reconstruye correctamente las 13 fuentes normalizadas, pero entrega sólo **4/13 traducciones castellanas aceptables y 9/13 fallbacks completos ingleses seguros**. El diagnóstico posterior de orden JSON entrega 8/13 y cinco fallbacks en ambos brazos; reutiliza el inglés y tiene presupuesto nuevo de traducción, por lo que **no sustituye ese resultado end-to-end**. El cambio de orden se rechaza y producción permanece v14. La búsqueda Lens real sigue sin validarse; las cuatro fuentes del pase completo se recuperaron mediante replay local explícito.

## Dataset y protocolo

Se transcribieron visualmente **13 diapositivas reales de dos carruseles**, con **77 regiones editoriales** independientes. Las expectativas, hashes, cajas, ambigüedades y fuentes conocidas están en [tests/regression/expected.json](tests/regression/expected.json). Las fotos permanecen fuera del repositorio, en `../TikTokTraducidos`, o en `TIKTOK_REGRESSION_MEDIA`.

Las expectativas se fijaron antes de comparar algoritmos. Se corrigieron dos límites de anotación tras revisar las dimensiones de las fotografías; la procedencia está registrada en el dataset. No se generó ground truth con un modelo. La detección mide cobertura de las cajas manuales; no confunde una lectura incorrecta con una caja ausente. Texto de fondo y ecuaciones se registran aparte.

Máquina: Windows 11, Intel Family 6 Model 186, 16 procesadores lógicos, 15,70 GiB de RAM; inferencia por CPU. OCR comparado: Python 3.11.16, PaddleOCR 3.7.0, Paddle 3.2.0 y OpenCV 4.10.0 dentro del contenedor real, ocho hilos. El build limpio posterior usa Python 3.11.17 y conserva esos modelos, Paddle y OpenCV; elimina la coexistencia de dos distribuciones OpenCV. Ollama: 0.34.4, `num_gpu=0`. El modelo configurado inicialmente era **Gemma 3 4B**, aunque la solicitud mencionaba aproximadamente Qwen 2.5 3B.

Una ejecución por configuración sobre todas las fotos, sin inferencias simultáneas. OCR y end-to-end calculan p95 por rango más próximo (`ceil(0,95*n)`); con 13 muestras equivale al máximo. La calibración de candidatos usa el percentil interpolado de NumPy y conserva esa definición en sus cifras publicadas. Los tiempos de OCR de la tabla excluyen inicialización y descarga de pesos. RSS es el máximo del proceso completo aislado, incluidas bibliotecas y modelos; no equivale al tamaño de los pesos.

## OCR: comparación medida

Detección correcta requiere al menos 50% de cobertura de cada caja manual. El reconocimiento compara las lecturas que caen geométricamente en esas cajas; nunca selecciona una frase por parecerse al texto esperado. CER y exactitud normalizadas omiten mayúsculas, puntuación y espacios.

| Configuración | Regiones detectadas | Diapositivas exactas normalizadas | CER | Media | p95 | Máximo RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BEFORE: PP-OCRv5 mobile detector + recognizer multilingüe | 77/77 | 9/13 | 0,003791 | 1,144 s | 2,343 s | 1.697 MiB |
| PP5 mobile detector + recognizer específico inglés | 77/77 | 13/13 | 0 | 0,841 s | 1,302 s | 1.621 MiB |
| PP-OCRv6 tiny | 77/77 | 9/13 | 0,003791 | 0,496 s | 0,763 s | 1.477 MiB |
| PP-OCRv6 small | 77/77 | 13/13 | 0 | 0,978 s | 1,461 s | 1.800 MiB |

**Decisión:** conservar Paddle y el detector PP5 mobile; usar `en_PP-OCRv5_mobile_rec`. En este dataset mejora cuatro diapositivas y reduce un 26,5% el tiempo medio frente al reconocedor multilingüe. PP6 tiny es más rápido pero conserva errores; PP6 small iguala la calidad léxica con mayor tiempo y memoria. No se añadieron dependencias de un segundo motor.

El resultado 13/13 no significa transcripción literal perfecta: siguen `waysto`, `opposite.Force` sin espacio y dos guiones de lista ausentes. El contenido léxico se conserva, incluida la errata impresa `Recources`. Estas diferencias se revisaron sobre las lecturas crudas.

La inicialización del modelo ya cacheado tardó 8,326 s. Inglés, tiny y small necesitaron descargar pesos por primera vez y sus inicializaciones fueron 68,086, 41,314 y 120,093 s: **no son tiempos de arranque comparables**. Se solicitó expresamente la descarga con `--allow-downloads`.

Experimento previo de preprocessings con el reconocedor antiguo: primary 9/13, native crop 10/13, grayscale 10/13 y CLAHE 9/13. Aplicar variantes siempre podía empeorar otras lecturas. Se implementó una sola pasada principal, hasta seis recortes nativos únicamente para regiones editoriales de confianza baja y una pasada grayscale sólo ante señales de cobertura/confianza sospechosas.

**Configuración adaptativa final `adaptive-gray-native-v4`:** 77/77 regiones, 13/13 diapositivas exactas normalizadas, CER 0 y cero diapositivas visibles vacías. Media **1,022 s**, p95/máximo **1,787 s**, mínimo **0,364 s**. No se activó la segunda pasada grayscale en estas 13 fotos; se conservan sus tests para casos sospechosos. Los símbolos repetitivos de baja confianza de los cables ya no provocan recortes y grayscale inútiles: Tesla pasó de 3,708 s en el ensayo intermedio a 1,383 s en esta ejecución. Son mediciones únicas, sujetas a variación de carga; no un intervalo estadístico.

Evidencia local: `.cache/regression/ocr-candidates.json` y, bajo la raíz de medios, `.cache/regression/ocr-before.json` y `ocr-after.json`. Se excluyen de Git los informes crudos que contienen rutas locales y los medios.

## Reconstrucción y traducción

Se compara por carrusel, con los mismos IDs y lecturas OCR medidas. La reconstrucción usa todas las líneas de cada diapositiva. Se separan generaciones crudas de traducciones realmente entregadas: una respuesta generada que después aborta no cuenta como éxito.

| Implementación / modelo | Entregadas | Reconstrucción exacta normalizada | Cláusulas entregadas / proxy | Tiempo de ambos carruseles | Pico RSS Ollama y runner |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEFORE / Gemma 3 4B | 7/13 | 10/13 | 32/81 | 186,763 s | 4,47 GB |
| BEFORE / Qwen 2.5 3B | 7/13 | 9/13 | 26/81 | 207,529 s | 2,38 GB |
| AFTER compacto intermedio / Gemma 3 4B | 13/13 | 12/13 | 79/81 | 186,755 s | 4,14 GB |

Estas filas reciben exactamente las mismas lecturas OCR antiguas, por carrusel; no se atribuye al modelo la mejora del recognizer inglés posterior. La última fila precede las guardas semánticas y correcciones descubiertas en el end-to-end, y **no certifica la versión final**. CER de reconstrucción pasó de 0,007582 a 0,000758; la secuencia de palabras coincidió en 8/13 frente a 11/13. El RSS muestreado suma frontend/runner y puede contar páginas compartidas más de una vez; GB es decimal.

Un ensayo intermedio con prompts más largos dio Gemma 3: 333,270 s, 12/13 reconstrucciones, 74/81 cláusulas, un fallback inglés; Qwen 3 4B instruct: 389,502 s, 11/13, 60/81 y dos fallbacks. Son los mismos casos y lecturas, pero distinto prompt del compacto: no sirven para afirmar una diferencia de velocidad entre familias bajo configuración idéntica. No justificaron sustituir Gemma.

El carrusel B abortaba por falsos nombres propios como `(Full` o por cognados españoles como `ideas` considerados inglés sin traducir. Se corrigieron esas guardas y un fallo de una diapositiva ya no elimina todas las demás.

La exactitud de reconstrucción mide caracteres y secuencia de palabras. Los checks de traducción contabilizan cláusulas de referencia, con alternativas explícitas: son un **proxy limitado**, no una certificación semántica. Un texto puede contener palabras correctas y aun invertir el significado; un test demuestra ese límite. La revisión manual se registra separadamente.

TranslateGemma 4B recibió las mismas reconstrucciones medidas de Gemma: 13 respuestas crudas, 66/81 cláusulas y 5/13 diapositivas con todos los checks, en 96,694 s **sólo de traducción**, RSS 3,86 GB. No pasó por las guardas de entrega de producción; entregadas/tiempo completo no son comparables con las filas anteriores. La revisión encontró cambio de persona y pérdida de «escribió sobre ello», añadidos sin evidencia y gramática incorrecta. No demostró calidad mayor o igual; se conserva la arquitectura textual con Gemma.

Marian EN→ES / CTranslate2 int8 se midió después sobre **esas mismas 13 reconstrucciones**, segmentadas en frases ya reconstruidas, no líneas OCR: 13 respuestas crudas, 76/81 cláusulas; **1,703 s total**, 0,298 s inicialización y 1,284 s generación, RSS muestreado **160,9 MiB**. Media por diapositiva 0,099 s, p95 0,153 s, mínimo 0,040 s. No pasó las guardas de entrega. La revisión encontró primera persona indebida, calcos, gramática y la lista Discord parcialmente en inglés; **no se adopta sólo por velocidad**.

Un ensayo separado en los seis casos difíciles con reconstrucciones actuales obtuvo 34/44 cláusulas y 2,852 s al pasar toda la diapositiva; se omitieron frases. Dividir únicamente frases completas reconstruidas obtuvo 40/44 y 1,974 s, conservando más contenido pero manteniendo fallos de persona y naturalidad. Esas seis entradas difieren del input OCR antiguo del benchmark de 13; sus tasas no se mezclan. Los informes son `text-marian-probes.json`, `text-marian-sentences-probes.json` y `text-marian-all13.json` dentro de `.cache/regression/`.

Qwen 2.5 7B, ya instalado, se ensayó por CPU sobre los seis casos difíciles actuales: reconstrucción exacta normalizada 6/6, **6/6 fallback inglés**, 542,664 s y RSS 5,165 GB. Declaró spans de incertidumbre pero los tradujo en vez de conservarlos; los controles rechazaron esas respuestas. Hubo además truncación y agotamiento del presupuesto. Esa ejecución descubrió que el límite antiguo de tokens no contemplaba los dos campos del esquema nuevo; el límite se corrigió y se conservó el fallo anterior, sin atribuirle una calidad de traducción inexistente ni cambiar el modelo por defecto. Evidencia: `text-qwen7b-before-budget-fix.json` y snapshot local de la implementación.

La arquitectura híbrida experimental reutilizó las seis reconstrucciones inglesas medidas y los borradores Marian por frases, y pidió a Gemma 3 4B revisarlos contra el inglés con las guardas de producción. Terminó en **292,098 s sólo de revisión**, RSS **4,39 GB**, 6/6 resultados y 33/44 cláusulas proxy. No incluye tiempo de reconstrucción ni generación del borrador: ambos se reutilizaron, sin inventar mediciones. A03/A04 terminaron en fallback inglés completo por declarar fragmentos inciertos y traducirlos; A03 además conservaba primera persona indebida en sus respuestas crudas. El holdout conservó `wyll` y tradujo el cuerpo, con fallback parcial previsto. B04 mantuvo el calco «Duerme sobre las decisiones» y B05 interpretó el título sin declarar incertidumbre, con «Enseña a alguien más simplificado». **Arquitectura rechazada y no integrada**: no demostró calidad suficiente ni una ventaja de coste. El informe `text-hybrid-revision-probes.json` conserva respuestas crudas, resultado entregado, guardas y flags; este ensayo precede el guard comparativo final v9.

La revisión manual del compacto encontró errores que el proxy no detectaba: pérdida de relación «about», identidad cambiada por posesión, calcos españoles y una interpretación no declarada del título ambiguo. Los controles de persona y relación detuvieron parte de esos errores en el end-to-end. Surgieron además «tu primer tenso» para «your first thought» y una corrección injustificada de la abreviatura entre comillas del holdout. La aceptación semántica sigue abierta mientras se corrigen y se repiten las pruebas reales; no se comunica 79/81 como porcentaje de traducción fiel.

La revisión bilingüe del primer end-to-end clasificó las 13 diapositivas de regresión en **7 traducciones naturales/fieles, 1 fallback inglés seguro y explícito, y 5 fallos de significado, naturalidad o incertidumbre no declarada**. El holdout editorial falló por `wyll → will`; su foto sin overlay no se puntúa. El desglose literal se conserva en `.cache/regression/e2e/pipeline-pre-quality-fix-manual-review.json`. El juicio manual y el proxy tienen criterios distintos y se muestran separados.

Los modelos reciben todas las líneas de la diapositiva, nunca imágenes ni ground truth como si fuera OCR. El reflujo une cláusulas partidas sin reescribir palabras y conserva títulos, nombres, listas y líneas protegidas. Las citas de un solo token se conservan literalmente; una cita de confianza OCR alta evita la reconstrucción generativa de esa diapositiva. Un encabezado inicial completo con imperativo de comunicación, destinatario indefinido y comparativo final sin objeto/conector se conserva en inglés con `AMBIGUOUS_COMPARATIVE_ATTACHMENT`. La regla es sintáctica y general: no coincide con contenido explícito ni preposiciones, no consulta el dataset y no selecciona una interpretación del título.

Los marcadores protegidos deben mantener identidad y orden exactos en reconstrucción y traducción. El esquema ES declara fragmentos inciertos y exige conservarlos literalmente; rechaza fuentes inventadas y fragmentos declarados que luego se traducen. La confianza OCR/alineación sigue siendo operacional y separada de ES: la traducción queda `UNVERIFIED`, sin probabilidad numérica ni certificación de equivalencia semántica, y los fallbacks tienen confianza LOW. En esa revisión v9 pasaron **100 tests de traducción**; incluían comparativos nuevos, contraejemplos con contenido/preposición, citas y marcadores. Son pruebas de contratos, no aceptación semántica del modelo; el conteo de la configuración más reciente figura en su revalidación posterior.

El presupuesto es de 180 s por llamada y 360 s por carrusel, con como máximo un reintento de cada diapositiva fallida. El límite de salida ahora contempla evaluación de fragmentos más destino completo: el caso de 723 caracteres y tres diapositivas tenía un cap de 457 tokens; el cálculo actualizado permite 1011. Se mantiene el máximo de 2800 y el contexto de 4096 tokens; los límites de tiempo siguen aplicándose. La telemetría registra caracteres de fuente, cap, tokens reales y causa de finalización. Qwen no se volvió a medir después de esa corrección y no se le atribuye una mejora hipotética.

Evidencia: `.cache/regression/text-before.json`, `text-after.json`, `text-after-first.json`, `text-specialized.json` y `text-guard-probes.json`. El ensayo inicial que agrupaba las 13 diapositivas en una sola llamada está marcado `comparison_eligible=false` en `text-all13-diagnostic.json` y no entra en la comparación BEFORE/AFTER.

## Recuperación de imágenes

El candidato público de Tesla era un falso negativo: coincidencia visual 0,965, 982 inliers y correlación 0,9886, rechazado por diagonales de la bobina detectadas como texto. La primera corrección del gate lo aceptó con el recognizer multilingüe. Al cambiar al inglés se midió otra regresión: los cables se reconocían como letras/símbolos y volvía a rechazarse. El gate actual conserva toda la evidencia y distingue símbolos diagonales de baja confianza de palabras normales o etiquetas cortas fiables.

Calibración real con los mismos originales, máscaras y CLEAN existentes, cuatro fuentes verificadas y variantes de crop/escala/JPEG: **12 positivos y 13 negativos**. El ensayo inglés anterior al arreglo dio **9 TP / 3 FN / 0 FP**; los tres falsos negativos eran Tesla. AFTER da **12 TP / 0 FN / 0 FP**. Media de validación **1,093 s**, p95 **1,986 s**; total 33,235 s incluye 4,957 s de la comparación adicional de modos. RSS máximo 1.058,9 MiB. Se mantiene el umbral **0,84**, sin bajarlo para forzar positivos. Conjunto pequeño y sin corpus independiente de calibración: no extrapolar esos cero falsos positivos a Internet.

En 13 candidatos que alcanzaron el gate, OCR adaptativo y primario tuvieron las mismas detecciones y decisiones. Excluyendo inicialización fría, 12 pares tuvieron media 0,781 s frente a 0,399 s. Otra comparación de 13 pares obtuvo decisiones idénticas con seis recortes frente a ninguno; 12 pares calientes bajaron de 0,547 a 0,367 s. SOURCE usa por ello una sola pasada principal y cero recortes; las originales conservan fallback adaptativo.

La comprobación de esa configuración final `masked-orb-v6-primary-source-no-crops` volvió a dar **12 TP / 0 FN / 0 FP**, media 0,937 s, p95 1,753 s, RSS 1.090,4 MiB. Total 28,937 s incluye 4,566 s de comparación de modos. Evidencia: `.cache/recovery-diagnosis/calibration-after.json` y `calibration-primary-crop-check.json` bajo la raíz de medios. Esas calibraciones mantienen los mismos bytes de máscaras/CLEAN anteriores para comparar; el end-to-end comprueba además máscaras recién generadas. Validar candidatos conocidos no demuestra encontrarlos con Lens.

Lens real tenía dos fallos: el file input aparecía tarde y el selector genérico elegía un adjunto de AI Mode en lugar del formulario de Lens. Ambos se reprodujeron y se corrigieron con tests. Tras la corrección se llega a **CAPTCHA** en Google, también con Chrome visible. No se intenta saltarlo. Recuperación por descubrimiento Lens de todo el dataset: **pendiente**, sin denominador ni tasa inventados.

La estrategia configurada es CLEAN completa → ORIGINAL completa → dos crops sin texto y con detalle visual → búsqueda textual. Se valida cada ronda y sólo continúan las diapositivas pendientes. Un candidato fiable detiene las consultas de esa diapositiva; CAPTCHA detiene la automatización y deja el motivo explícito. Si existen candidatos rechazados, el resultado conserva `NEEDS_REVIEW`, score, razones e intentos. Un presupuesto total configurable, 300 s por defecto, impide continuar consultas y esperas vencidas; se registra `TIMEOUT`, nunca un `NOT_FOUND` que finja haber agotado búsquedas.

## End-to-end y rendimiento completo

Se descargó un tercer carrusel público real, `7553302113757990166`, con dos fotos, en 3,4 s de gallery-dl; 5,2 s para el probe inicial. Se revisaron ambas visualmente antes del OCR. [tests/regression/holdout.json](tests/regression/holdout.json) conserva tres líneas editoriales en la primera y marca la segunda como retrato sin texto editorial superpuesto, con logos de fondo. La abreviatura `wyll` debe mantenerse literalmente; no se inventa su expansión. Este holdout separado no modifica el denominador de 13 usado en BEFORE/AFTER.

Los tres carruseles completaron descarga real, OCR, CLEAN, matcher/gate y llamadas Ollama, seguidos de repetición con caché, en una salida aislada. Las cuatro fuentes verificadas quedaron `MATCH` tras la segunda ronda; ese fallback fue **replay offline de candidatos conocidos**, no descubrimiento Lens ni descarga HTTP. `MEASURED` en el informe indica ejecución/integridad/cache, no aceptación de traducción. Se detectaron los fallos semánticos descritos arriba y se preservan para la revisión siguiente.

Durante esa ejecución quedaron sólo 63,8 MiB de RAM disponible en Windows. Los tiempos completos bajo esa presión no constituyen una comparación controlada con BEFORE ni con el benchmark OCR aislado. La segunda foto del holdout sólo tiene logos: su OCR crudo se conserva y no se puntúa como una diapositiva con texto editorial perdido.

| Post / cohorte | Fotos | Descarga | OCR | CLEAN | Validación/replay | Reconstrucción | Traducción/modelo | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7659483315203804449 / conocido | 7 | 4,899 s | 17,196 s | 6,042 s | 12,308 s | 42,708 s | 89,522 s | 175,634 s |
| 7682913895333383456 / mucho texto | 6 | 16,590 s | 34,115 s | 29,897 s | 30,491 s | 156,162 s | 149,233 s | 420,650 s |
| 7553302113757990166 / nuevo, poco texto | 2 | 8,367 s | 15,923 s | 2,000 s | 3,562 s | 29,255 s | 50,984 s | 112,644 s |

Media por carrusel **236,309 s**, p95/máximo **420,650 s**, mínimo **112,644 s**, n=3. El total incluye preparación/guardado además de las fases de la tabla. Peak worker 1.576,3 MiB; peak Ollama/llama-server 3.837,5 MiB, muestreados por separado. No se puede obtener un BEFORE completo comparable de manifests históricos con caché/errores y condiciones distintas; se deja **N/A** y se conserva la comparación controlada por componentes.

Los probes de caché fueron 3,939 / 7,103 / 5,347 s, con todos los contadores de trabajo a cero. Incluyen la ejecución y la instantánea posterior de comprobación de archivos; no son una medición aislada de la latencia de `run_pipeline`. El informe preservado está en `.cache/regression/e2e/pipeline-pre-quality-fix.json`, con resumen, log y muestreo de memoria; no se sobrescribe como prueba de la versión corregida.

El manifest registra descarga, OCR principal/fallback, CLEAN, cada consulta, descarga/comparación/OCR de candidatos, reconstrucción, traducción y total. Los tiempos de llamadas por lote se comparten entre sus diapositivas, sin atribuirles una medición individual ficticia.

### Revalidación v9 de las copias existentes

Después de los arreglos de citas, encabezados ambiguos y máscaras se ejecutaron nuevamente los 15 casos sobre copias de la evidencia anterior, con código `evidence-slide-fallback-v9`. Se bloquearon descarga, resolución de URL y transferencia de candidatos. Se reutilizó el OCR medido y se regeneraron CLEAN/máscaras; las cuatro SOURCE se revalidaron con el matcher y OCR actuales. Por eso estos tiempos **no son procesamiento fresco completo ni una mejora de velocidad comparable con la tabla anterior**.

Los 13 casos principales tuvieron 77/77 regiones detectadas, reconocimiento y reconstrucción 13/13 normalizados. El holdout editorial tuvo 3/3 regiones, reconocimiento y reconstrucción exactos; la foto de logos permanece fuera del denominador editorial. Cuatro SOURCE conservaron hashes y punteros y pasaron de nuevo el gate. Las nuevas consultas fueron cero; los cuatro negativos rechazados pertenecen a la evidencia anterior, no se volvieron a ofrecer aquí.

Tiempos del pipeline por carrusel: 172,093 / 238,105 / 129,556 s, media 179,918 s, p95/máximo 238,105 s, mínimo 129,556 s, n=3. El bucle del worker duró 547,991 s e incluye posts y comprobaciones; la clonación se mide aparte. Los tres probes `OK_CACHED` tuvieron latencia del pipeline 0,382 / 0,513 / 0,345 s; ejecución más hash 0,921 / 0,993 / 0,731 s. Todos sus contadores de trabajo adicional fueron cero. Permanecieron intactos 77 archivos de la evidencia anterior y los 46 artefactos del usuario protegidos.

La revisión bilingüe v9 clasificó el corpus principal en **8 traducciones naturales/fieles, 2 fallbacks ingleses seguros y 3 fallos**: cambio de objeto en `self-sabotage`, calco `Sin entrada` en un contexto cerebral y locución `Duerme sobre las decisiones`. El holdout conserva ahora literalmente `wyll`, pero entrega toda la frase en inglés por rechazar una declaración redundante del marcador. Esa conservación segura no se cuenta como una traducción española exitosa. Las notas de estilo y errores literales están en `.cache/regression/e2e/v9-pre-lexical-fix-manual-review.json`; los informes, logs y hashes v9 se preservan con el mismo prefijo. **v9 no superó la aceptación semántica** y justifica la siguiente corrección acotada.

### Revalidación v10 y diagnóstico del formato

Se conservaron las copias y métricas v9 antes de repetir los mismos 15 casos con `evidence-slide-fallback-v10`. Tres guardas acotadas rechazaron cambio de objeto en `self-sabotage`, la locución deliberativa `sleep on` traducida como dormir sobre algo y `no input` como entrada física en contexto cerebral. El reintento aporta únicamente una definición inglesa contextual, no la traducción esperada. La declaración redundante de un marcador ya protegido sólo se acepta si coincide literalmente y conserva cantidad y orden.

La revisión manual v10 encontró **8 traducciones naturales/fieles, 5 fallbacks completos ingleses seguros y ninguna traducción española incorrecta conocida entregada** en el corpus principal. El holdout conservó `wyll`, pero dejó toda la frase en inglés por fallo de integridad del marcador. Esto mejora la seguridad respecto a v9, pero **no demuestra aceptación completa de traducción ni imposibilidad de traducir los textos legibles restantes**. Los fallos persistentes son A03/A04/B02/B04/B05; sus razones están en `.cache/regression/e2e/v10-pre-schema-fix-manual-review.json`.

Detección y reconocimiento/reconstrucción principales: 77/77 regiones y 13/13 normalizados, CER 0; holdout editorial 3/3 regiones y 1/1 exacto. Las cuatro SOURCE pasaron de nuevo matcher/OCR, sin nuevas búsquedas, transferencias ni OCR de originales. Siempre se generaron los 15 CLEAN. Se conservaron los 77 archivos de la evidencia anterior y los 46 artefactos protegidos del usuario. Los negativos pertenecen a la prueba anterior y no se volvieron a medir.

Tiempos del pipeline por carrusel: **150,843 / 312,903 / 132,066 s**, media **198,604 s**, p95/máximo **312,903 s**, mínimo **132,066 s**, n=3. Son revalidación de copias con OCR reutilizado, no procesamiento fresco ni comparación causal de velocidad. El bucle del worker duró 602,832 s, sin incluir clonación ni transporte del host. Los tres `OK_CACHED` tuvieron latencia del pipeline 0,327 / 0,410 / 0,193 s, con todos los contadores adicionales a cero; los probes con hash tardaron 1,145 / 0,785 / 0,454 s. RAM disponible mínima 526,34 MiB, peak Ollama 3.661,29 MiB y worker 1.015,93 MiB.

La suite v10 pasó **324 tests, 1 warning, 41,85 s**, incluidos 111 de traducción. El launcher recreó sólo el traductor en 8,611 s, conservó el hash de la imagen y comprobó API/health antes de guardar la nueva firma. Informes, logs, memoria, tests, reinicio e inventario SHA v10 permanecen preservados bajo `.cache/regression/e2e/v10-pre-schema-fix*`.

Un probe separado de cuatro casos difíciles v10 tardó 242,660 s y obtuvo reconstrucción 4/4 normalizada. A04/B02/B04 quedaron completos en inglés; el holdout sí produjo castellano con `wyll` literal. Esa diferencia frente al lote completo muestra sensibilidad al agrupamiento y no permite certificarlo con el probe. El experimento siguiente restringió los spans declarados a subcadenas exactas del DATA de cada diapositiva y añadió ejemplos genéricos de contrato, sin frases del dataset ni flexibilizar las guardas semánticas; el conjunto se rechazó tras la revalidación v12.

### Experimento v11: source fundamentado y comparación equivalente

El esquema experimental limitó `uncertain_spans.source` a un máximo de 256 subcadenas literales del DATA de **ese mismo ID**, con cobertura del principio/final y sin marcadores parciales. Tres ejemplos genéricos enseñaron diferencias entre ID y número de título, marcador literal y término desconocido; no contenían frases del dataset. La documentación de [Ollama](https://docs.ollama.com/capabilities/structured-outputs) y el [convertidor JSON Schema de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) respaldan el uso de enum. La restricción no garantiza que el destino copie el span ni preserve significado; no permanece en la configuración v13.

Se midieron **las mismas seis entradas OCR de producción**, con sus confidencias y IDs, agrupadas en sus tres posts. Cinco proceden del corpus principal y una del holdout; este subconjunto no reemplaza el denominador de 13. Ambos modelos se descargaron de memoria antes de arrancar por CPU, sin inferencias simultáneas y sin descargar pesos nuevos.

| Modelo / mismo código v11 | Inglés entregado exacto normalizado | Castellano fiel con fragmento literal | Inglés completo seguro | Castellano con fallo manual | Tiempo total de texto | RSS Ollama/runner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 4B | 6/6 | 1/6 | 3/6 | 2/6 | 369,849 s | 4,303 GB |
| Qwen 2.5 3B | 6/6 | 1/6 | 4/6 | 1/6 | 258,844 s | 2,384 GB |

En Gemma, B02 contiene OCR literal exacto porque falló la reconstrucción del modelo por numeración: **6/6 de fidelidad del inglés entregado no significa seis reconstrucciones del modelo aceptadas**. Gemma preservó correctamente el encabezado ambiguo B05, pero escribió `detrás del jerga`; B04 perdió el consejo de aplazar y dejar pasar una noche mediante `Reflexiona sobre las decisiones`. Qwen cambió ese consejo por soñar y añadió fallos gramaticales. El cuerpo del holdout es fiel en ambos, con `wyll` literal. Los proxies fueron 27/44 y 20/44 respectivamente; ni siquiera la presencia de palabras correctas detectó estos errores semánticos.

**Decisión:** no sustituir Gemma por Qwen sólo por velocidad; ambos fallan la aceptación del subconjunto. Los informes, SHA y revisión se preservan en `.cache/regression/text-schema-v11*`. El experimento justifica dos guardas más acotadas: conservar el consejo temporal del encabezado formal `sleep on` y comprobar determinante masculino junto a `jerga`, sustantivo femenino según el [DLE de RAE/ASALE](https://dle.rae.es/jerga). No se precarga una traducción española ni se afirma validación gramatical universal. También justifica permitir traducción desde OCR literal de evidencia alta cuando la respuesta de reconstrucción fue rechazada por contrato, conservando ese rechazo como diagnóstico separado.

### Revalidación v12: experimento rechazado

El pase completo v12 conservó el paquete enum + tres ejemplos y añadió las correcciones deterministas de presupuesto/diagnóstico, OCR literal tras rechazos de contenido, encabezado temporal, concordancia de `jerga` e idempotencia de marcadores. Se repitieron los 15 casos desde una nueva copia del mismo snapshot, sin descarga, búsqueda ni OCR de originales adicionales.

La revisión bilingüe principal encontró **6/13 traducciones aceptables, 6/13 fallbacks completos ingleses seguros y 1/13 fallo de naturalidad**. A04 entregó `Te auto-saboteas cada buena cosa de tu vida`: conserva el objeto explícito y la acepción de autosabotaje, pero la construcción es forzada. A05/A06, aceptables en v10, retrocedieron a inglés por fallo de copia literal. El holdout sí tradujo fielmente el cuerpo y conservó `"wyll"`; eso no compensa la regresión principal. La revisión completa está en `.cache/regression/e2e/v12-final-manual-review.json`.

**Decisión:** retirar conjuntamente enum y ejemplos, restaurando schema/prompt v10 y manteniendo las correcciones deterministas. No se atribuye causalidad aislada a uno de los componentes sin una ablación. La configuración v13 añade sólo una definición inglesa en el reintento existente de `write about`, para distinguir tema de escritura y objeto escrito; no contiene castellano esperado ni entradas de ground truth.

OCR/reconstrucción principal: 77/77 regiones y 13/13 normalizados, CER 0; holdout editorial 3/3 y 1/1. Las cuatro SOURCE pasaron el gate actual; 15 CLEAN, cero intentos de red, 77 archivos anteriores y 46 del usuario intactos. Los negativos no se volvieron a medir. La rama nueva de traducción desde OCR literal **no se ejercitó en este pase real**; todos los textos entregados tenían fuente `RECONSTRUCTED`.

Pipeline por carrusel: 205,566 / 291,774 / 121,249 s; media **206,196 s**, p95/máximo **291,774 s**, mínimo **121,249 s**, n=3. Revalidación con OCR reutilizado, no ejecución fresca. Bucle del worker 629,957 s; clonación/hash 17,126 s aparte. Caché: 0,409 / 1,461 / 0,180 s de pipeline, sin trabajo adicional; probes con hash 1,709 / 2,174 / 0,523 s. RAM disponible mínima 484,23 MiB; Ollama peak 3.994,79 MiB y worker 1.021,96 MiB.

La suite v12 pasó **351 tests, 1 warning, 41,48 s**, incluidos 138 de traducción. Restart sólo translator: 9,665 s, sin build, misma imagen, API/health, firma y SHA del módulo comprobados. Nueve evidencias y los hashes de 77 artefactos se preservaron bajo `.cache/regression/e2e/v12-final*`; v9/v10 permanecen intactos. La revisión integrada v13 está en curso y no se certifica anticipadamente.

### Revalidación v13: mejora de cobertura y defecto localizado

Tras retirar el paquete experimental se repitieron los mismos 15 casos. Las ocho salidas automáticas `TRANSLATED` **no equivalen a ocho aceptaciones de naturalidad**: la revisión independiente encontró **7/13 castellanos aceptables, 5/13 fallbacks completos ingleses seguros y B03 pendiente de naturalidad**. `Oblígate a que tu cerebro defienda ambos lados` conserva el sentido y el objeto, pero la construcción resulta forzada. El holdout tradujo correctamente el cuerpo y conservó `"wyll"`. No se interpreta este hallazgo como pérdida semántica ni prohibición universal de reflexivos.

Se recuperaron A05/A06 respecto al experimento rechazado. OCR/reconstrucción principal 77/77 regiones y 13/13 normalizados, CER 0; holdout editorial 3/3 y 1/1. Siempre hubo 15 CLEAN y cuatro SOURCE revalidadas, con cero búsquedas/transferencias, cero nuevos OCR de originales, 77 archivos de v7 y 46 del usuario intactos. Los negativos siguen siendo evidencia anterior, sin nueva medición; la continuación desde OCR literal no se ejercitó.

Pipeline por carrusel: **225,767 / 270,432 / 118,017 s**; media **204,739 s**, p95/máximo **270,432 s**, mínimo **118,017 s**, n=3. Son revalidación con OCR reutilizado, no procesamiento fresco. Bucle del worker 624,209 s; clonación/hash 14,514 s aparte. La ventana del monitor fue 642,904 s e incluye staging/clonación/worker/transferencia, con resolución de dos segundos; no se registró un timer exacto del wrapper completo. Caché: 0,360 / 0,606 / 0,311 s de pipeline, todos los contadores extra cero; probes con hash 0,900 / 1,050 / 0,949 s. RAM disponible mínima 551,27 MiB; peak Ollama/llama-server 3.710,89 MiB y worker 1.015,06 MiB, con scopes separados.

Suite **348 tests, 1 warning, 41,52 s**, incluidos 135 de traducción; wrapper de tests 42,500 s. Restart sólo translator 9,834 s sin build, misma imagen, firma v13 y SHA del módulo comprobados. Nueve evidencias y 77 hashes se preservaron bajo `.cache/regression/e2e/v13-final*`, junto a la lectura bilingüe; v9/v10/v12 permanecen intactos.

La corrección v14 añade únicamente una comprobación localizada de la forma causativa forzada, con glosa inglesa en el reintento existente y conservación de inglés si persiste. La aceptación v14 requiere nueva inferencia; no se certifica anticipadamente.

### Revalidación v14: B03 corregida, cobertura global insuficiente

El control de naturalidad se ejercitó en B03 y el reintento produjo `Obligue a su cerebro a defender ambas partes`, natural y fiel en un registro coherente de usted. La revisión independiente principal encontró **4/13 castellanos aceptables y 9/13 fallbacks completos ingleses seguros**, sin castellano incorrecto conocido. A07 omite `¿` inicial: defecto ortográfico real, separado de significado y naturalidad. A01–A06 fallaron repetidamente el contrato de copiar un source declarado válido, aunque el inglés reconstruido es correcto. La guarda nueva sólo afecta B03; **este descenso global no demuestra causalidad de la guarda**. El holdout vuelve a pasar con cuerpo español y `"wyll"` literal.

Permanecen 77/77 regiones y 13/13 OCR/reconstrucción normalizados, CER 0; holdout editorial 3/3 y 1/1. Siempre 15 CLEAN, cuatro SOURCE revalidadas, cero intentos de red o transferencia, 77 archivos de v7 y 46 del usuario intactos. Los negativos no se volvieron a medir; ninguna de las 15 fuentes de texto ejercitó la continuación desde OCR literal.

Pipeline por carrusel: **269,804 / 268,459 / 131,789 s**; media **223,351 s**, p95/máximo **269,804 s**, mínimo **131,789 s**, n=3. Revalidación con OCR reutilizado, no ejecución fresca. Bucle del worker 678,379 s; clonación/hash 12,244 s aparte. Ventana del monitor 694,541 s con resolución de dos segundos, sin timer exacto del wrapper. Caché: 0,392 / 0,652 / 0,140 s de pipeline, todos los contadores extra cero; probes con hash 0,762 / 1,120 / 0,395 s. RAM disponible mínima 457,75 MiB; peak Ollama/llama-server 3.905,23 MiB y worker 1.032,65 MiB.

Suite **357 tests, 1 warning, 41,60 s**, incluidos 144 de traducción; wrapper de tests 42,624 s. Restart sólo translator 9,609 s sin build, misma imagen, salud API2, firma v14 y SHA del módulo comprobados. Nueve evidencias y 77 hashes preservados bajo `.cache/regression/e2e/v14-final*`; v9/v10/v12/v13 intactos. La revisión bilingüe está en `v14-final-manual-review.json`.

El diagnóstico histórico demuestra que un `source` válido faltaba literalmente en `spanish`; las respuestas rechazadas de ese pase no se capturaron. Por tanto no permite distinguir traducción, omisión o alteración del fragmento. La nota manual anterior se conservó y se precisó, sin modificar las mediciones archivadas.

### Diagnóstico pareado del orden JSON: cambio rechazado

Se compararon `metadata-first` y `spanish-first` sobre **14 diapositivas editoriales, grupos 7+6+1**; la foto nueva de logos se excluyó por su anotación previa. Ambas variantes usan exactamente el mismo inglés medido v14, protecciones y DATA, Gemma 3 4B Q4_K_M, validadores, límites y opciones. Sólo cambian el orden de propiedades/requeridos y las dos directivas coherentes de secuencia. Cada post/brazo dispone de **360 s nuevos sólo para ES**, sin OCR ni reconstrucción. Una ejecución serial alterna el primer brazo entre posts; no hay repeticiones ni una comparación directa con el presupuesto residual end-to-end.

| Variante | Castellano aceptable principal | Inglés completo seguro | Holdout editorial | ES observado A / B / holdout | Suma de jobs |
| --- | ---: | ---: | --- | --- | ---: |
| metadata-first | 8/13 | 5/13 | Cuerpo fiel, `wyll` literal | 89,821 / 203,061 / 13,915 s | 306,797 s |
| spanish-first | 8/13 | 5/13 | FAIL naturalidad: comilla y coma sueltas finales | 139,412 / 225,095 / 23,833 s | 388,340 s |

La lectura bilingüe independiente acepta A01/A02/A05/A06/A07/B01/B03/B06 en ambos brazos. A03/A04/B02/B04/B05 quedan completamente en inglés. El formato inverso respeta realmente su orden en **19/19 objetos de ocho llamadas**, y el control respeta el suyo en **20/20 objetos de nueve llamadas**; ningún DATA cambió. Esto descarta que el engine impusiera un orden contrario en esta medición. El patrón histórico de copia fallida en A01–A06 no se reproduce ni siquiera en el control; no se atribuye retrospectivamente al orden.

Las capturas nuevas localizan otros fallos: cuatro `source` inventados a partir de etiquetas de reparación en el control; un ID JSON usado como `source` y un marcador declarado pero omitido en el destino inverso. Los validadores rechazan esas respuestas. En ambos brazos persisten errores de actor/relación, autodestrucción reflexiva, sentido de `input`, consejo temporal o encabezado protegido; no son ambigüedad demostrada de todo el texto ni evidencia de imposibilidad de traducirlo. El holdout inverso añade literalmente `",` al final del cuerpo: el JSON exterior es válido y termina por `stop`, pero el texto entregado no pasa la revisión estricta de naturalidad.

**Decisión: no adoptar spanish-first.** No mejora cobertura entregada y añade ruido de salida. Los tiempos observados tampoco prueban un efecto causal de rendimiento: las cachés de evaluación de prompt fueron desiguales y sólo se midió un par por post. Total del instrumento **696,641 s**, wrapper **697,323 s**, RSS máximo Ollama **4.339,36 MiB**. La producción v14 y sus hashes permanecieron intactos. Evidencia local: `.cache/regression/text-schema-order.json`, `text-schema-order-summary.json`, `text-schema-order-execution.json` y `text-schema-order-manual-review.json`; se excluye `thinking`.

La suite integrada posterior pasa **364 tests, 1 warning, 42,13 s**, wrapper **43,180 s**, sin fallos, errores ni skips. Añade siete pruebas de preparación/captura del instrumento y conserva los 144 tests de traducción. Es evidencia de contratos en el venv Windows; no una nueva inferencia ni un reemplazo del pase15 v14. Los resultados están en `.cache/regression/schema-order-final-tests.*`.

## Reproducción

```powershell
.\setup.ps1 -CheckOnly
.\.venv\Scripts\python.exe tools/benchmark_ocr_candidates.py --allow-downloads
.\tools\run_ocr_benchmark.ps1 -Adaptive -Strict
.\.venv\Scripts\python.exe tools/benchmark_text_regression.py --help
.\.venv\Scripts\python.exe tools/benchmark_marian.py --help
.\.venv\Scripts\python.exe tools/benchmark_recovery.py --help
.\.venv\Scripts\python.exe tools/debug_slide.py 7659483315203804449 3
```

Los benchmarks OCR y end-to-end fallan si faltan originales o sus hashes no coinciden. La herramienta de recuperación ya incluye un preflight equivalente antes de importar OCR, más existencia de SOURCE e identidad del ORIGINAL seleccionado desde el manifest; pasó sobre los 13 originales y cuatro fuentes reales. Las calibraciones publicadas preceden ese preflight: se compararon posteriormente sus 25 hashes de original por informe contra el dataset, sin diferencias. Las fuentes de referencia se revisaron visualmente, pero no tienen un hash esperado fijado en el dataset. `.cache/robustness-before` conserva el código y manifests anteriores localmente; no se publica. La reproducción AFTER requiere las fotos referenciadas. La comparación exacta BEFORE requiere además ese snapshot local.

## Alternativas actuales investigadas

La [documentación oficial de PaddleOCR](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) describe los modelos pequeños medidos. [EasyOCR](https://github.com/JaidedAI/EasyOCR) permite CPU pero añade PyTorch; [docTR](https://github.com/mindee/doctr) usa una arquitectura de dos etapas. [Surya 2](https://github.com/datalab-to/surya) utiliza un VLM y servidor de inferencia, por lo que no cumple aquí la restricción de LLM sólo textual. No se instalaron ni se les atribuyen resultados locales.

Se revisaron [Qwen 3.5](https://ollama.com/library/qwen3.5), [Gemma 4](https://ollama.com/library/gemma4), [TranslateGemma](https://huggingface.co/google/translategemma-4b-it), [Marian EN→ES](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) y [CTranslate2](https://opennmt.net/CTranslate2/guides/transformers.html). Las opciones no ejecutadas sólo son investigación; no justifican cambiar la arquitectura ni recibir una puntuación.

El ensayo adicional de Marian usa [un checkpoint convertido a CTranslate2 int8](https://huggingface.co/playto-mt/opus-mt-en-es-ct2), revisión `a363bfbf3a34a116bb50ca32fedf7379338eea8b`. El archivo de 68.502.783 bytes coincide con el SHA-256 LFS publicado `addfb36ac165b121d5f38a85fb8b3a122fecf66040732f14eed619c279bfa210`. Se extrajeron sólo archivos/directorios seguros en caché; no se ejecuta código remoto. CTranslate2 4.8.2 y SentencePiece 0.2.1 se instalaron en una caché aislada de benchmark; no forman parte de las dependencias de producción. [tools/benchmark_marian.py](tools/benchmark_marian.py) recibe pesos locales y un informe de reconstrucciones medidas; no descarga ni sustituye entradas por ground truth. La revisión experimental Gemma sobre ese borrador terminó y se rechazó por calidad y coste; [tools/benchmark_translation_revision.py](tools/benchmark_translation_revision.py) conserva el protocolo reproducible sin integrarlo en producción.
