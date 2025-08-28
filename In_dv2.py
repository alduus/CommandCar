#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae domicilios desde PostgreSQL, mueve el prefijo de tipo de vía desde 'calle' a 'tipo_via'
SIN canonizar, deja 'calle' tal cual cuando no hay prefijo y exporta a CSV.

Uso típico:
  python normalizar_domicilios.py \
      --tabla domicilios \
      --schema public \
      --limit 10 \
      --outdir salidas \
      --salida prueba.csv

Producción (todo el dataset):
  python normalizar_domicilios.py --tabla domicilios --schema public --limit 0
"""

import os
import re
import csv
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql
from dotenv import load_dotenv, find_dotenv

# -------------------- Config base --------------------

# Nombres lógicos (como quieres trabajarlos en pandas/CSV)
TARGET_COLS = [
    "tipo_via",
    "calle",
    "numero_exterior",
    "numero_interior",
    "colonia",
    "municipio",
    "ciudad",
    "estado",
    "codigo_postal",
]

# Si en tu BD algunos nombres reales son distintos, mapéalos aquí:
#   clave = nombre lógico (arriba), valor = nombre REAL en la BD
# Ejemplo: en tu captura se ve "tipo_vía" con acento
REAL_COL_MAP = {
    "tipo_via": "tipo_vía",   # cambialo si en tu BD NO lleva acento
    # Agrega más mapeos si tienes diferencias:
    # "codigo_postal": "código_postal",
}

# Prefijos detectables en 'calle' (solo para detección; NO se normaliza el texto)
VARIANT_KEYS = {
    "calle", "cll",
    "avenida", "av", "av.",
    "bulevar", "bulevard", "boulevard", "blvd", "blvd.",
    "circuito", "cto", "cto.",
    "calzada", "calz", "calz.", "czda",
    "camino", "cno",
    "privada", "priv", "priv.", "pvda",
    "prolongacion", "prolongación", "prol", "prol.",
    "carretera", "carr", "cte",
    "autopista", "aut",
    "andador", "and.",
    "pasaje", "pje", "psje",
    "callejon", "callejón", "cljon",
    "eje",
    "periferico", "periférico", "perif",
    "anillo",
}

try:
    from unidecode import unidecode
except Exception:
    def unidecode(x):
        return x

def _build_prefix_pattern():
    keys = sorted({unidecode(k.lower()) for k in VARIANT_KEYS}, key=len, reverse=True)
    keys_escaped = [re.escape(k) for k in keys]
    return re.compile(r"^\s*(?:" + r"|".join(keys_escaped) + r")(?=\s|\.?\s|$)", flags=re.IGNORECASE)

PREFIX_RE = _build_prefix_pattern()

# -------------------- Normalización fila a fila --------------------

def extract_type_from_calle_raw(calle_original: str):
    """
    Si 'calle_original' empieza con un tipo de vía, devuelve (tipo_via_crudo, resto_calle_tal_cual).
    Si no, devuelve (None, calle_original sin cambios).
    """
    if not isinstance(calle_original, str) or not calle_original.strip():
        return None, calle_original

    calle_no_acento = unidecode(calle_original)
    m = PREFIX_RE.search(calle_no_acento)
    if not m:
        return None, calle_original.strip()

    start, end = m.span()
    prefijo_en_original = calle_original[: (end - start)].strip()
    prefijo_en_original = re.sub(r"\.$", "", prefijo_en_original.strip())

    resto = calle_original[(end - start):]
    resto = re.sub(r"^\s*\.?\s*", "", resto)

    return prefijo_en_original, resto.strip()

def normalize_row(row: dict) -> dict:
    calle = row.get("calle")
    tipo_actual = row.get("tipo_via")
    nuevo_tipo, nueva_calle = extract_type_from_calle_raw(calle)

    if nuevo_tipo:
        row["tipo_via"] = nuevo_tipo          # se mueve tal cual
        row["calle"] = nueva_calle            # resto tal cual
    else:
        row["tipo_via"] = tipo_actual         # sin cambio
        row["calle"] = calle                  # sin cambio
    return row

# -------------------- Conexión y lectura segura --------------------

def load_env_file(path_env: str | None):
    if path_env:
        ok = load_dotenv(path_env, override=False)
        if ok:
            print(f".env cargado desde: {path_env}")
        else:
            print(f"Advertencia: no se pudo cargar .env desde {path_env}")
    else:
        found = find_dotenv()
        load_dotenv(found, override=False)
        if found:
            print(f".env detectado y cargado: {found}")
        else:
            print("No se encontró .env; se usarán variables de entorno del sistema.")

def connect_pg_from_env():
    params = {
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
    }
    missing = [k for k, v in params.items() if v in (None, "")]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)} (define en .env)")
    return psycopg2.connect(
        host=params["host"],
        port=params["port"],
        dbname=params["dbname"],
        user=params["user"],
        password=params["password"],
        cursor_factory=RealDictCursor,
    )

def _real_name(logical: str) -> str:
    """Devuelve el nombre REAL de la columna en la BD según REAL_COL_MAP."""
    return REAL_COL_MAP.get(logical, logical)

def leer_dataframe(conn, tabla: str, cols, limit: int = 0, schema: str | None = None, debug_sql: bool = True):
    """
    SELECT robusto con psycopg2.sql:
    - Respeta esquema si se indica.
    - Usa alias: SELECT "real" AS "lógico".
    - Nunca usa literales en las columnas.
    """
    # Limpia nombres lógicos por si traen comillas erróneas
    cols = [c.strip().strip('"').strip("'") for c in cols]

    # Validar existencia contra information_schema con el nombre REAL
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
              AND (%s IS NULL OR table_schema = %s)
            """,
            (tabla, schema, schema),
        )
        existentes = {r["column_name"] for r in cur.fetchall()}

    real_needed = [_real_name(c) for c in cols]
    faltantes = [r for r in real_needed if r not in existentes]
    if faltantes:
        raise RuntimeError(f"La tabla '{tabla}' no tiene las columnas requeridas (nombres reales): {faltantes}")

    # SELECT "real" AS "logico"
    fields = []
    for logical in cols:
        real = _real_name(logical)
        fields.append(
            sql.SQL("{} AS {}").format(sql.Identifier(real), sql.Identifier(logical))
        )

    table_ident = sql.Identifier(tabla) if not schema else sql.SQL(".").join([sql.Identifier(schema), sql.Identifier(tabla)])

    query = sql.SQL("SELECT {fields} FROM {table}").format(
        fields=sql.SQL(", ").join(fields),
        table=table_ident,
    )
    if limit and limit > 0:
        query = query + sql.SQL(" LIMIT {}").format(sql.Literal(int(limit)))

    query_str = query.as_string(conn)

    # Guardias: si por alguna razón aparecen comillas simples en columnas, aborta
    if "SELECT '" in query_str:
        raise RuntimeError(f"SQL inválido (literal en SELECT): {query_str}")

    if debug_sql:
        print("SQL =>", query_str)

    # Cargar DataFrame
    df = pd.read_sql(query_str, conn)

    # Sanity check: si la primera fila son los nombres de columnas, indica SELECT mal formado
    if len(df) > 0 and df.iloc[0].astype(str).tolist() == cols:
        raise RuntimeError("Los datos parecen literales (primera fila igual a nombres de columnas). Revisa SELECT.")

    return df

# -------------------- Exportación --------------------

def exportar_csv(df: pd.DataFrame, ruta_salida: Path):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"Archivo generado: {ruta_salida}")

# -------------------- CLI --------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Extrae domicilios desde PostgreSQL, mueve prefijo de vía desde 'calle' a 'tipo_via' sin canonizar y exporta a CSV."
    )
    p.add_argument("--tabla", default="domicilios", help="Tabla origen (default: domicilios)")
    p.add_argument("--schema", default=None, help="Esquema de la tabla (ej. public)")
    p.add_argument("--env", default="", help="Ruta a archivo .env (default: autodetect)")
    p.add_argument("--limit", type=int, default=10, help="Filas a leer para prueba (0=todas)")
    p.add_argument("--outdir", default="", help="Carpeta de salida (default: OUTPUT_DIR de .env o ./salidas)")
    p.add_argument("--salida", default="", help="Nombre del CSV (default: domicilios_normalizados_YYYY_MM_DD_HhMm.csv)")
    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Cargar .env
    load_env_file(args.env if args.env else None)

    # Carpeta de salida
    base_out = args.outdir.strip() or os.getenv("OUTPUT_DIR", "salidas")
    outdir = Path(base_out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Nombre del archivo
    nombre = args.salida.strip()
    if not nombre:
        ts = datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
        nombre = f"domicilios_normalizados_{ts}.csv"
    salida_path = outdir / nombre

    # Conexión y lectura
    conn = connect_pg_from_env()
    try:
        print("TARGET_COLS =>", TARGET_COLS)
        df = leer_dataframe(
            conn=conn,
            tabla=args.tabla,
            cols=TARGET_COLS,
            limit=args.limit,
            schema=args.schema,
            debug_sql=True,
        )
        print(f"Registros leídos: {len(df)}")

        # Normalización
        df = df.apply(normalize_row, axis=1)

        # Exportación
        exportar_csv(df, salida_path)
    finally:
        conn.close()

if __name__ == "__main__":
    main()