# Taller NLP — Agente 10-K

Documentación detallada del baseline (perfiles de retrieval, configuración
congelada, etc.) en [`common/README.md`](common/README.md). Esta sección
cubre solo lo necesario para ejecutar el proyecto desde un clon limpio.

## Ejecución desde un clon limpio

1. **Clonar** (asegúrate de estar en la rama `integration/final-agent`,
   donde vive el código final; `master` no la incluye):

   ```powershell
   git clone https://github.com/CristianCamilo98/taller_nlp.git
   cd taller_nlp
   git checkout integration/final-agent
   ```

2. **Crear y activar el entorno** (Python 3.11 o 3.12):

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias**:

   ```powershell
   python -m pip install -r common\requirements.txt
   # Opcional, para tests/desarrollo/notebook (incluye ipykernel):
   python -m pip install -r common\requirements-dev.txt
   ```

4. **Colocar el dataset**. No está en Git (`.gitignore`). Debe existir,
   con esta estructura, en `dataset/` junto al repo (o dentro de él), o en
   cualquier ruta indicada con `MIAX_DATASET_DIR`:

   ```powershell
   $env:MIAX_DATASET_DIR = 'C:\ruta\a\tu\dataset'
   ```

   Dentro de esa carpeta deben existir `corpus_miax_2026/` (con
   `secciones.jsonl`, `chunks.jsonl`, `xbrl_facts.parquet`) y
   `indice_faiss_gemini_embedding_2/` (con `corpus.faiss`,
   `chunks_meta.parquet`, `index_manifest.json` — el índice usado por el
   perfil de retrieval final). Ver [`common/README.md`](common/README.md#dataset)
   para el resto de perfiles.

5. **Configurar la API key de OpenRouter**. Copia la plantilla y rellena tu
   clave real (nunca se sube a Git; `.env` está en `.gitignore`):

   ```powershell
   Copy-Item .env.example .env
   # Edita .env y pon tu clave real en OPENROUTER_API_KEY
   ```

6. **Ejecutar el notebook de entrega**: abre
   [`notebook_entrega.ipynb`](notebook_entrega.ipynb) (raíz del repo) y
   ejecútalo de arriba a abajo. No reconstruye índices ni lanza baterías de
   preguntas automáticas.

7. **Ejemplo `responder(pregunta)`** (llamada real a OpenRouter):

   ```python
   from common.responder import responder
   respuesta = responder("¿Cuál fue el total de activos (Assets) de NVIDIA en el ejercicio fiscal 2024?")
   ```

8. **Ejemplo `evaluar(ruta_jsonl)`** sobre un archivo propio de preguntas
   (cada línea un JSON con, como mínimo, `id`, `familia`, `pregunta` — mismo
   esquema que `common/golden_set/golden_set_grupo3.jsonl`; acepta cualquier
   número de preguntas nuevas, no requiere IDs conocidos de antemano):

   ```python
   from common.evaluar import evaluar
   resultados = evaluar("ruta/a/tus_10_preguntas.jsonl", guardar_en="resultados.jsonl")
   ```

9. **Ejecutar la suite de tests** (offline, sin llamadas a red ni API key):

   ```powershell
   python -m unittest discover -s common\tests -v
   ```
