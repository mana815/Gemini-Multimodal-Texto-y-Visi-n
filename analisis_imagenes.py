from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "imagenes"
RESULTS_PATH = BASE_DIR / "resultados_imagenes.json"

IMAGENES = [
    {
        "id": 1,
        "archivo": "documento_escaneado.jpg",
        "tipo": "documento escaneado",
        "tarea_principal": "OCR",
        "esperado": {
            "ocr": "FACTURA DEMO Cliente Ana López Empresa Tarraco Tech Fecha 29/09/2026 Concepto Servicio de mantenimiento Subtotal 35,12 EUR IVA 7,38 EUR TOTAL 42,50 EUR Gracias por su confianza",
            "escena": ["documento", "factura"],
            "objetos": ["documento", "texto"],
            "conceptos_descripcion": ["factura", "cliente", "total"],
        },
    },
    {
        "id": 2,
        "archivo": "calle_urbana.jpg",
        "tipo": "foto urbana",
        "tarea_principal": "clasificación de escena y objetos",
        "esperado": {
            "ocr": "CENTRO",
            "escena": ["calle urbana", "ciudad", "urbana"],
            "objetos": ["coche", "edificio", "semáforo", "paso de peatones"],
            "conceptos_descripcion": ["calle", "coches", "edificios", "semáforo"],
        },
    },
    {
        "id": 3,
        "archivo": "producto_cafe.jpg",
        "tipo": "producto",
        "tarea_principal": "OCR y descripción de producto",
        "esperado": {
            "ocr": "CAFÉ SIERRA 100% ARÁBICA 250 g",
            "escena": ["producto", "producto sobre mesa", "bodegón"],
            "objetos": ["bolsa de café", "taza"],
            "conceptos_descripcion": ["café", "bolsa", "taza"],
        },
    },
    {
        "id": 4,
        "archivo": "restaurante.jpg",
        "tipo": "interior",
        "tarea_principal": "descripción y detección de objetos",
        "esperado": {
            "ocr": "MENÚ",
            "escena": ["restaurante", "comedor", "interior de restaurante"],
            "objetos": ["pizza", "mesa", "vaso", "menú"],
            "conceptos_descripcion": ["pizza", "mesa", "restaurante"],
        },
    },
    {
        "id": 5,
        "archivo": "circuito_formula1.jpg",
        "tipo": "deporte de motor",
        "tarea_principal": "clasificación de escena y objetos",
        "esperado": {
            "ocr": "32 META RACE",
            "escena": ["circuito", "carrera", "automovilismo", "fórmula 1"],
            "objetos": ["coche de carreras", "pista", "grada"],
            "conceptos_descripcion": ["coche", "circuito", "carrera"],
        },
    },
]

SCHEMA = {
    "type": "object",
    "properties": {
        "ocr_texto": {"type": "string"},
        "descripcion": {"type": "string"},
        "escena": {"type": "string"},
        "objetos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "cantidad_aproximada": {"type": "integer", "minimum": 1},
                },
                "required": ["nombre", "cantidad_aproximada"],
                "additionalProperties": False,
            },
        },
        "confianza": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["ocr_texto", "descripcion", "escena", "objetos", "confianza"],
    "additionalProperties": False,
}


def obtener_sdk_google():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "No se encuentra google-genai. Instala: python -m pip install -U google-genai Pillow"
        ) from exc
    return genai, types


def norm(texto: str) -> str:
    tabla = str.maketrans("áéíóúüñ", "aeiouun")
    return " ".join(texto.casefold().translate(tabla).replace("/", " ").split())


def similitud_ocr(esperado: str, obtenido: str) -> float:
    return SequenceMatcher(None, norm(esperado), norm(obtenido)).ratio()


def acierto_escena(esperadas: list[str], obtenida: str) -> float:
    ob = norm(obtenida)
    return 1.0 if any(norm(x) in ob or ob in norm(x) for x in esperadas) else 0.0


def recall_lista(esperados: list[str], obtenidos: list[str]) -> float:
    if not esperados:
        return 1.0
    obs = [norm(x) for x in obtenidos]
    aciertos = 0
    for e in esperados:
        ne = norm(e)
        if any(ne in x or x in ne for x in obs):
            aciertos += 1
    return aciertos / len(esperados)


def cobertura_descripcion(conceptos: list[str], descripcion: str) -> float:
    texto = norm(descripcion)
    if not conceptos:
        return 1.0
    return sum(1 for c in conceptos if norm(c) in texto) / len(conceptos)


def comparar(item: dict[str, Any], resultado: dict[str, Any]) -> dict[str, float]:
    esperado = item["esperado"]
    objetos = [x["nombre"] for x in resultado.get("objetos", [])]
    return {
        "ocr": round(similitud_ocr(esperado["ocr"], resultado.get("ocr_texto", "")), 4),
        "descripcion": round(cobertura_descripcion(esperado["conceptos_descripcion"], resultado.get("descripcion", "")), 4),
        "escena": round(acierto_escena(esperado["escena"], resultado.get("escena", "")), 4),
        "objetos": round(recall_lista(esperado["objetos"], objetos), 4),
    }


def analizar_real(client: Any, types: Any, item: dict[str, Any]) -> dict[str, Any]:
    path = IMAGE_DIR / item["archivo"]
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        datos = buffer.getvalue()

    prompt = f"""
Analiza esta imagen de test ({item["tipo"]}).

Haz cuatro tareas:
1. OCR: extrae literalmente el texto visible que puedas leer. Si no hay texto, usa "".
2. Descripción: describe el contenido visual de forma clara y concreta.
3. Clasificación de escena: devuelve una etiqueta breve para el tipo de escena.
4. Detección de objetos: enumera los objetos relevantes y una cantidad aproximada.

La tarea prioritaria de esta imagen es: {item["tarea_principal"]}.
Devuelve ÚNICAMENTE JSON conforme al esquema.
""".strip()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=datos, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            temperature=0.2,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini ha devuelto una respuesta vacía.")

    return json.loads(response.text)


def demo_para(item: dict[str, Any]) -> dict[str, Any]:
    e = item["esperado"]
    nombres = e["objetos"]
    return {
        "ocr_texto": e["ocr"],
        "descripcion": (
            "[DEMO] Imagen de "
            + item["tipo"]
            + " con "
            + ", ".join(e["conceptos_descripcion"])
            + "."
        ),
        "escena": e["escena"][0],
        "objetos": [{"nombre": x, "cantidad_aproximada": 1} for x in nombres],
        "confianza": 0.95,
    }


def procesar_demo() -> list[dict[str, Any]]:
    salida = []
    for item in IMAGENES:
        resultado = demo_para(item)
        salida.append({
            **item,
            "resultado": resultado,
            "precision": comparar(item, resultado),
        })
    return salida


def procesar_real() -> list[dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No existe GEMINI_API_KEY.")

    genai, types = obtener_sdk_google()
    client = genai.Client(api_key=api_key)

    salida = []
    for i, item in enumerate(IMAGENES, start=1):
        print(f"[{i:02d}/5] Procesando {item['archivo']}...")
        resultado = analizar_real(client, types, item)
        salida.append({
            **item,
            "resultado": resultado,
            "precision": comparar(item, resultado),
        })
        if i < len(IMAGENES):
            time.sleep(6)
    return salida


def metricas_globales(items: list[dict[str, Any]]) -> dict[str, float]:
    claves = ["ocr", "descripcion", "escena", "objetos"]
    medias = {}
    for clave in claves:
        medias[clave] = round(
            sum(x["precision"][clave] for x in items) / len(items), 4
        )
    medias["media_global"] = round(sum(medias[k] for k in claves) / len(claves), 4)
    return medias


def guardar(modo: str, resultados: list[dict[str, Any]]) -> dict[str, Any]:
    data = {
        "modo": modo,
        "modelo": MODEL,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "numero_imagenes": len(resultados),
        "metricas": metricas_globales(resultados),
        "resultados": resultados,
    }
    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def imprimir(data: dict[str, Any]) -> None:
    print("\n=== ANÁLISIS DE IMÁGENES CON GEMINI ===")
    print(f"Modo: {data['modo'].upper()}")
    print(f"Modelo: {data['modelo']}")
    print(f"Imágenes: {data['numero_imagenes']}")
    print("\nTabla comparativa de precisión:")
    print("-" * 76)
    print(f"{'Imagen':26} {'OCR':>10} {'Descripción':>13} {'Escena':>10} {'Objetos':>10}")
    print("-" * 76)
    for item in data["resultados"]:
        p = item["precision"]
        print(
            f"{item['archivo'][:26]:26} "
            f"{p['ocr']*100:9.1f}% "
            f"{p['descripcion']*100:12.1f}% "
            f"{p['escena']*100:9.1f}% "
            f"{p['objetos']*100:9.1f}%"
        )
    print("-" * 76)
    m = data["metricas"]
    print(
        f"{'MEDIA':26} "
        f"{m['ocr']*100:9.1f}% "
        f"{m['descripcion']*100:12.1f}% "
        f"{m['escena']*100:9.1f}% "
        f"{m['objetos']*100:9.1f}%"
    )
    print(f"\n[OK] Guardado: {RESULTS_PATH.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    try:
        resultados = procesar_demo() if args.demo else procesar_real()
        data = guardar("demo" if args.demo else "real", resultados)
        imprimir(data)
        return 0

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        texto = str(exc).lower()
        if "403" in texto or "permission_denied" in texto or "denied access" in texto:
            print(
                "La API está rechazando el proyecto de Google AI Studio. "
                "Documenta el 403/estado Restringido y usa --demo para comprobar el flujo local.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
