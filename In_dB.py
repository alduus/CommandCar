#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lee la tabla 'domicilios' en PostgreSQL, mueve el prefijo de tipo de vía desde 'calle' a 'tipo_via'
SIN canonizar y deja 'calle' tal cual (sin title-case si no hay prefijo).
"""

import os
import re
import csv
import argparse
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from unidecode import unidecode
except Exception:
    def unidecode(x):
        return x

# --- Columnas a trabajar ---
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

# --- Variantes conocidas (solo para detección, no para normalizar) ---
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

def _build_prefix_pattern():
    keys = sorted({unidecode(k.lower()) for k in VARIANT_KEYS}, key=len, reverse=True)
    keys_escaped = [re.escape(k) for k in keys]
    return re.compile(r"^\s*(?:" + r"|".join(keys_escaped) + r")(?=\s|\.?\s|$)", flags=re.IGNORECASE)

PREFIX_RE = _build_prefix_pattern()

def extract_type_from_calle_raw(calle_original: str):
    """
    Si 'calle_original' empieza con un tipo de vía,
    devuelve (tipo_via_crudo, resto_calle_tal_cual).
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
        row["tipo_via"] = nuevo_tipo
        row["calle"] = nueva_calle
    else:
        # No prefijo: se deja todo tal cual
        row["tipo_via"] = tipo_actual
        row["calle"] = calle
    return row

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
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")
    return psycopg2.connect(
        host=params["host"],
        port=params["port"],
        dbname=params["dbname"],
        user=params["user"],
        password=params["password"],
        cursor_factory=RealDictCursor,
    )

def leer_dataframe(conn, tabla: str, cols):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (tabla,),
        )
        existentes = {r["column_name"] for r in cur.fetchall()}
    faltantes = [c for c in cols if c not in existentes]
    if faltantes:
        raise RuntimeError(f"La tabla '{tabla}' no tiene las columnas requeridas: {faltantes}")
    col_list = ", ".join(cols)
    df = pd.read_sql(f'SELECT {col_list} FROM "{tabla}"', conn)
    return df

def exportar_csv(df: pd.DataFrame, ruta_salida: str):
    df.to_csv(ruta_salida, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

def main():
    parser = argparse.ArgumentParser(description="Mueve prefijo de tipo de vía desde 'calle' a 'tipo_via' sin canonizar ni modificar 'calle'.")
    parser.add_argument("--tabla", default="domicilios")
    parser.add_argument("--salida", default="")
    args = parser.parse_args()

    salida = args.salida.strip()
    if not salida:
        ts = datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
        salida = f"domicilios_normalizados_{ts}.csv"

    conn = connect_pg_from_env()
    try:
        df = leer_dataframe(conn, args.tabla, TARGET_COLS)
        print(f"Registros leídos: {len(df)}")
        df = df.apply(normalize_row, axis=1)
        exportar_csv(df, salida)
        print(f"Archivo generado: {salida}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
