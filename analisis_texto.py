from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Continuación directa de gemini_practica.py del ejercicio anterior.
# El modelo se puede cambiar sin editar el código:
# PowerShell -> $env:GEMINI_MODEL="gemini-3.8-flash"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "resultados_texto.json"

TEXTOS = [
    {
        "id": 1,
        "tipo": "noticia",
        "texto": "Microsoft anunció en Madrid una herramienta de inteligencia artificial para ayudar a pequeñas empresas a automatizar tareas administrativas.",
        "esperado": {"idioma": "es", "sentimiento": "neutro", "entidades_clave": ["Microsoft", "Madrid"]},
    },
    {
        "id": 2,
        "tipo": "reseña",
        "texto": "El restaurante fue excelente. La comida llegó rápido, el personal fue muy amable y volvería sin dudarlo.",
        "esperado": {"idioma": "es", "sentimiento": "positivo", "entidades_clave": ["restaurante"]},
    },
    {
        "id": 3,
        "tipo": "email",
        "texto": "Hola Laura, la reunión con Acme se ha movido al jueves a las 11:00 en Barcelona. Confírmame si puedes asistir.",
        "esperado": {"idioma": "es", "sentimiento": "neutro", "entidades_clave": ["Laura", "Acme", "Barcelona"]},
    },
    {
        "id": 4,
        "tipo": "reseña",
        "texto": "Estoy muy decepcionado con el portátil. Se apaga solo, la batería dura poco y el servicio técnico no resolvió el problema.",
        "esperado": {"idioma": "es", "sentimiento": "negativo", "entidades_clave": ["portátil", "batería", "servicio técnico"]},
    },
    {
        "id": 5,
        "tipo": "noticia",
        "texto": "Renfe informó que el nuevo servicio entre Tarragona y Barcelona comenzará el próximo lunes con cuatro frecuencias diarias.",
        "esperado": {"idioma": "es", "sentimiento": "neutro", "entidades_clave": ["Renfe", "Tarragona", "Barcelona"]},
    },
    {
        "id": 6,
        "tipo": "noticia",
        "texto": "Apple presented a new accessibility feature in London that will be included in a future software update.",
        "esperado": {"idioma": "en", "sentimiento": "neutro", "entidades_clave": ["Apple", "London"]},
    },
    {
        "id": 7,
        "tipo": "reseña",
        "texto": "I absolutely love this camera. The image quality is fantastic and the autofocus is fast and reliable.",
        "esperado": {"idioma": "en", "sentimiento": "positivo", "entidades_clave": ["camera", "autofocus"]},
    },
    {
        "id": 8,
        "tipo": "email",
        "texto": "Hi Daniel, your flight to Paris has been changed to 18:30 on Friday. Please check the airline app for the updated boarding gate.",
        "esperado": {"idioma": "en", "sentimiento": "neutro", "entidades_clave": ["Daniel", "Paris", "Friday"]},
    },
    {
        "id": 9,
        "tipo": "reseña",
        "texto": "The delivery was three days late and the box arrived damaged. Customer support was not helpful at all.",
        "esperado": {"idioma": "en", "sentimiento": "negativo", "entidades_clave": ["delivery", "Customer support"]},
    },
    {
        "id": 10,
        "tipo": "email",
        "texto": "Gracias por enviar el informe. El contenido está claro y bien organizado, aunque sería útil añadir una gráfica al final.",
        "esperado": {"idioma": "es", "sentimiento": "positivo", "entidades_clave": ["informe", "gráfica"]},
    },
]

SCHEMA = {
    "type": "object",
    "properties": {
        "idioma": {"type": "string", "enum": ["es", "en", "otro"]},
        "entidades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "texto": {"type": "string"},
                    "tipo": {"type": "string"},
                },
                "required": ["texto", "tipo"],
                "additionalProperties": False,
            },
        },
        "sentimiento": {"type": "string", "enum": ["positivo", "negativo", "neutro"]},
        "confianza": {"type": "number", "minimum": 0, "maximum": 1},
        "resumen": {"type": "string"},
    },
    "required": ["idioma", "entidades", "sentimiento", "confianza", "resumen"],
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


def normalizar(valor: str) -> str:
    return " ".join(valor.casefold().split())


def calcular_precision(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(resultados)
    idiomas_ok = 0
    sentimientos_ok = 0
    entidades_total = 0
    entidades_ok = 0

    for item in resultados:
        esperado = item["esperado"]
        obtenido = item["resultado"]

        idiomas_ok += int(obtenido["idioma"] == esperado["idioma"])
        sentimientos_ok += int(obtenido["sentimiento"] == esperado["sentimiento"])

        detectadas = [normalizar(e["texto"]) for e in obtenido.get("entidades", [])]
        for entidad in esperado["entidades_clave"]:
            entidades_total += 1
            objetivo = normalizar(entidad)
            if any(objetivo in x or x in objetivo for x in detectadas):
                entidades_ok += 1

    p_idioma = idiomas_ok / total if total else 0
    p_sent = sentimientos_ok / total if total else 0
    p_ent = entidades_ok / entidades_total if entidades_total else 0

    return {
        "idioma": round(p_idioma, 4),
        "sentimiento": round(p_sent, 4),
        "entidades_recall": round(p_ent, 4),
        "media_global": round((p_idioma + p_sent + p_ent) / 3, 4),
        "detalle": {
            "idioma": f"{idiomas_ok}/{total}",
            "sentimiento": f"{sentimientos_ok}/{total}",
            "entidades": f"{entidades_ok}/{entidades_total}",
        },
    }


def analizar_real(client: Any, types: Any, entrada: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""
Analiza este texto y devuelve ÚNICAMENTE JSON.

Tareas:
1. Detecta el idioma: "es", "en" u "otro".
2. Extrae entidades nombradas y elementos relevantes, indicando su tipo.
3. Clasifica el sentimiento: positivo, negativo o neutro.
4. Indica una confianza entre 0 y 1.
5. Resume el texto en una o dos frases.

Tipo de texto: {entrada["tipo"]}
Texto: {entrada["texto"]}
""".strip()

    respuesta = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            temperature=0.2,
        ),
    )

    if not respuesta.text:
        raise RuntimeError("Gemini ha devuelto una respuesta vacía.")

    return json.loads(respuesta.text)


def modo_demo() -> list[dict[str, Any]]:
    # Permite comprobar la práctica aunque AI Studio esté restringido.
    salida = []
    for entrada in TEXTOS:
        resultado = {
            "idioma": entrada["esperado"]["idioma"],
            "entidades": [
                {"texto": e, "tipo": "entidad"}
                for e in entrada["esperado"]["entidades_clave"]
            ],
            "sentimiento": entrada["esperado"]["sentimiento"],
            "confianza": 0.95,
            "resumen": f"[DEMO] Resumen automático del texto {entrada['id']}.",
        }
        salida.append({**entrada, "resultado": resultado})
    return salida


def modo_real() -> list[dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No existe GEMINI_API_KEY.")

    genai, types = obtener_sdk_google()
    client = genai.Client(api_key=api_key)

    salida = []
    for i, entrada in enumerate(TEXTOS, start=1):
        print(f"[{i:02d}/10] Analizando {entrada['tipo']}...")
        resultado = analizar_real(client, types, entrada)
        salida.append({**entrada, "resultado": resultado})
        if i < len(TEXTOS):
            time.sleep(6)
    return salida


def guardar(modo: str, resultados: list[dict[str, Any]]) -> dict[str, Any]:
    data = {
        "modo": modo,
        "modelo": MODEL,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "numero_textos": len(resultados),
        "metricas": calcular_precision(resultados),
        "resultados": resultados,
    }
    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    try:
        resultados = modo_demo() if args.demo else modo_real()
        data = guardar("demo" if args.demo else "real", resultados)

        print("\n=== ANÁLISIS AVANZADO DE TEXTO ===")
        print(f"Modo: {data['modo'].upper()}")
        print(f"Modelo: {data['modelo']}")
        print(f"Textos: {data['numero_textos']}")
        print(f"Precisión idioma: {data['metricas']['idioma']*100:.1f}%")
        print(f"Precisión sentimiento: {data['metricas']['sentimiento']*100:.1f}%")
        print(f"Recall entidades: {data['metricas']['entidades_recall']*100:.1f}%")
        print(f"[OK] Guardado: {RESULTS_PATH.name}")
        return 0

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        texto = str(exc).lower()
        if "403" in texto or "permission_denied" in texto or "denied access" in texto:
            print(
                "La llamada llega a Google, pero el proyecto de AI Studio está restringido. "
                "Documenta el 403 y usa --demo para comprobar el flujo local.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
