# Neurociencia 2026–2027

Repositorio con tres flujos de trabajo relacionados con el proyecto ATC Trust: una aplicación de adquisición y análisis en tiempo real, herramientas para exportar datos de la tarea y un preprocesador de eye tracking. Las grabaciones y los resultados exportados se conservan en el repositorio como datos del proyecto.

## Estructura

```text
.
├── ATCO_trust_project/            Aplicación multimodal en tiempo real y sus dependencias
├── ATC_EXTRACTOR/                 Extracción, validación y exportación de datos de la tarea
│   ├── raw_csv/                   Exportaciones originales de la tarea
│   ├── personality_csv/           Respuestas de cuestionarios
│   ├── scripts/                   Variantes de la tarea Trust ATC
│   └── output/                    Libros Excel exportados
├── Códigos para preprocesado/     Preprocesador de datos de eye tracking en Excel
├── docs/references/               Artículos y material de referencia
└── recordings/                    Sesiones registradas por la aplicación
```

`ATC_EXTRACTOR/rt_validation_*.csv` contiene informes de validación. Los datos de `recordings/`, los CSV de entrada y los Excel exportados son artefactos del análisis; no se ignoran automáticamente.

## Aplicación en tiempo real

Ejecuta los comandos desde la raíz del repositorio. La aplicación escribe las sesiones nuevas en `recordings/` mediante una ruta relativa al directorio actual.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ATCO_trust_project/requirements.txt
python ATCO_trust_project/main.py
```

## Extracción y validación

Los scripts del extractor usan rutas relativas (`raw_csv/`, `personality_csv/` y `output/`), así que ejecútalos desde `ATC_EXTRACTOR/`:

```bash
cd ATC_EXTRACTOR
python extractor.py
```

El procesamiento que incorpora cuestionarios está en `process_data_with_personality.py`; la validación está en `validator.py`. Las variantes de la tarea están en `scripts/`.

### Análisis H2: cambio de confianza

El análisis de H2 usa el libro consolidado `ATC_EXTRACTOR/datos_combinados.xlsx` y guarda el informe, las tablas y las figuras en `ATC_EXTRACTOR/output/h2_trust_change/`. Con la venv activa, instala las dependencias del análisis y ejecútalo desde la raíz:

```bash
python -m pip install -r ATC_EXTRACTOR/requirements-analysis.txt
python ATC_EXTRACTOR/h2_trust_analysis.py
```

El script compara el cambio de confianza entre ensayos consecutivos, dentro del mismo bloque, y clasifica el análisis principal por `correct` del ensayo actual. También incluye un análisis separado por `automation_error`, que describe si la sugerencia automática fue errónea.

## Preprocesamiento de eye tracking

Ejecuta `Códigos para preprocesado/Preprocesado_ET.py`. El programa permite seleccionar un libro Excel y guarda junto a él otro libro con los datos limpios, parpadeos y sacadas.

## Higiene del repositorio

`.gitignore` excluye metadatos de macOS, bytecode de Python y entornos virtuales. Los datos experimentales y los resultados exportados siguen versionados; revisa su contenido antes de añadir nuevas grabaciones o exportaciones.
