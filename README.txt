TikTok Translator local para Windows

La documentación actual está en README.md.

Desde esta carpeta:
  .\setup.ps1
  .\run.ps1

En una instalacion nueva, las descargas grandes requieren opciones explicitas:
  .\setup.ps1 -PullModels -BuildBackend
Chrome se reutiliza; para descargar Chromium: .\setup.ps1 -InstallBrowser.
Comprobacion sin instalar ni iniciar servicios: .\setup.ps1 -CheckOnly.

También puedes abrir launcher\start.cmd con doble clic.

Resultados predeterminados: ..\TikTokTraducidos\<POST_ID>\
Traducciones: traducciones.txt. Imágenes: original, clean y recovered.
Google Lens puede necesitar interacción en Chrome. Recovered contiene bytes web RAW; CLEAN queda separado.
Al terminar se pide otro enlace. Para cerrar: salir, exit o q.

Consulta README.md para requisitos, configuración y estado de verificación.
