# Análisis avanzado de texto e imágenes con Gemini

Este repositorio es la **continuación directa de la práctica anterior** (`gemini_practica.py`), en la que ya se trabajó con:

- `google-genai`
- conexión con `genai.Client(...)`
- prompts de texto
- análisis de sentimiento
- respuestas JSON
- una imagen multimodal
- guardado de resultados en JSON

En este ejercicio se amplía esa base con análisis avanzado de texto, OCR, clasificación visual, detección de objetos y un notebook de análisis.

## Estructura

```text
Gemini_Analisis_Avanzado/
├── analisis_texto.py
├── analisis_imagenes.py
├── analisis_resultados.ipynb
├── resultados_texto.json
├── resultados_imagenes.json
├── requirements.txt
├── README.md
├── .gitignore
└── imagenes/
    ├── documento_escaneado.jpg
    ├── calle_urbana.jpg
    ├── producto_cafe.jpg
    ├── restaurante.jpg
    └── circuito_formula1.jpg
```

## 1. Análisis avanzado de texto

`analisis_texto.py` procesa 10 textos de ejemplo (noticias, reseñas y emails).

Para cada texto pide a Gemini:

1. detección de idioma;
2. extracción de entidades;
3. análisis de sentimiento;
4. nivel de confianza;
5. resumen automático.

La respuesta se fuerza a JSON estructurado y se guarda en `resultados_texto.json`.

También se comparan valores esperados con los obtenidos y se calcula:

- precisión de idioma;
- precisión de sentimiento;
- recall de entidades;
- media global.

## 2. Análisis de imágenes

`analisis_imagenes.py` procesa 5 imágenes variadas.

Para cada imagen pide:

1. OCR;
2. descripción de contenido visual;
3. clasificación de escena;
4. detección de objetos.

Las cinco pruebas son:

| Imagen | Tipo | Tarea principal |
|---|---|---|
| `documento_escaneado.jpg` | documento | OCR |
| `calle_urbana.jpg` | urbana | escena + objetos |
| `producto_cafe.jpg` | producto | OCR + descripción |
| `restaurante.jpg` | interior | descripción + objetos |
| `circuito_formula1.jpg` | deporte | escena + objetos |

Los resultados se guardan en `resultados_imagenes.json`.

El script calcula una tabla comparativa de precisión para:

- OCR;
- cobertura de descripción;
- clasificación de escena;
- detección de objetos.

## 3. Notebook Jupyter

`analisis_resultados.ipynb` carga los dos JSON y genera:

- tabla de resultados de texto;
- tabla comparativa de precisión visual;
- resumen de métricas;
- gráfico comparativo;
- conclusión final.

## Requisitos

Python 3.13 o superior.

Instalación:

```powershell
python -m pip install -r requirements.txt
```

## Modelo

Por defecto:

```text
gemini-3.8-flash
```

Se puede cambiar sin editar los scripts:

```powershell
$env:GEMINI_MODEL="gemini-3.8-flash"
```

## API Key

No se escribe la clave dentro del código.

En una terminal de VS Code:

```powershell
$env:GEMINI_API_KEY="TU_API_KEY"
```

Después:

```powershell
python analisis_texto.py
python analisis_imagenes.py
```

## Modo demo

Durante la práctica se detectó que el proyecto de Google AI Studio aparecía como **Restringido** y la API devolvía:

```text
403 PERMISSION_DENIED
Your project has been denied access
```

Por este motivo, ambos scripts incluyen un modo `--demo` para poder comprobar localmente todo el flujo mientras el acceso real está bloqueado.

```powershell
python analisis_texto.py --demo
python analisis_imagenes.py --demo
```

El modo demo:

- no usa la API;
- genera los dos JSON;
- permite abrir y ejecutar el notebook;
- está marcado explícitamente como `"modo": "demo"`.

Para la entrega conviene documentar que los resultados demo son de comprobación local y que el intento real quedó bloqueado por el estado del proyecto de Google AI Studio.

## Abrir el notebook

```powershell
jupyter notebook analisis_resultados.ipynb
```

También se puede abrir directamente desde VS Code si está instalada la extensión Jupyter.

## Ejecución recomendada mientras la API está restringida

```powershell
python analisis_texto.py --demo
python analisis_imagenes.py --demo
jupyter notebook analisis_resultados.ipynb
```

## Relación con el ejercicio anterior

No se ha empezado desde cero. Se ha mantenido el mismo patrón técnico de la práctica anterior:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(...)
```

La diferencia es que ahora la práctica amplía el análisis a 10 textos, 5 imágenes, métricas de precisión y un informe Jupyter.

## Entregable

El repositorio incluye exactamente lo solicitado:

- scripts de texto y visión con Gemini;
- cinco imágenes de prueba;
- resultados JSON;
- notebook Jupyter con análisis de resultados.
