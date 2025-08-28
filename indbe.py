import os
import re
import csv
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv, find_dotenv

try:
    from unidecode import unidecode
except Exception:  # pragma: no cover
    def unidecode(x):
        return x

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
    "callejon", "callejón", "cljon",
    "eje",
    "anillo",
}

#aca nomas se detecta el prefijo
def _build_prefix_pattern():
    keys = sorted({unidecode(k.lower()) for k in VARIANT_KEYS}, key=len, reverse=True)
    keys_escaped = [re.escape(k) for k in keys]
    return re.compile(r"^\s*(?:" + r"|".join(keys_escaped) + r")(?=\s|\.?\s|$)", flags=re.IGNORECASE)

PREFIX_RE = _build_prefix_pattern()

#aca se extrae y se normaliza para que entre la regex
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

#aca respeta y regresa como viene en vdi
def normalize_row(row: dict) -> dict:
    """
    - Si 'calle' tiene prefijo de tipo de vía: lo mueve a 'tipo_via' tal cual y deja el resto en 'calle'.
    - Si no hay prefijo: no cambia nada (se respeta 'calle' y 'tipo_via' originales).
    """
    calle = row.get("calle")
    tipo_actual = row.get("tipo_via")
    nuevo_tipo, nueva_calle = extract_type_from_calle_raw(calle)

    if nuevo_tipo:
        row["tipo_via"] = nuevo_tipo
        row["calle"] = nueva_calle
    else:
        row["tipo_via"] = tipo_actual
        row["calle"] = calle
    return row


# ------------------ Conexión a Postgres ----------------
def connect_db():
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


def leer_dataframe(conn, tabla: str, cols, limit: int = 0, schema: str | None = None, debug_sql: bool = False):
    # Limpia nombres por si vinieran con comillas en la lista
    cols = [c.strip().strip('"').strip("'") for c in cols]

    # Verifica existencia de columnas
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

    faltantes = [c for c in cols if c not in existentes]
    if faltantes:
        raise RuntimeError(f"La tabla '{tabla}' no tiene las columnas requeridas: {faltantes}")

    identifiers = [sql.Identifier(c) for c in cols]
    table_ident = (
        sql.SQL(".").join([sql.Identifier(schema), sql.Identifier(tabla)])
        if schema else sql.Identifier(tabla)
    )

    query = sql.SQL("SELECT {fields} FROM {table}").format(
        fields=sql.SQL(", ").join(identifiers),
        table=table_ident,
    )
    if limit and limit > 0:
        query = query + sql.SQL(" LIMIT {}").format(sql.Literal(int(limit)))

    if debug_sql:
        print(query.as_string(conn))

    df = pd.read_sql(query.as_string(conn), conn)
    return df


def exportar_csv(df: pd.DataFrame, ruta_salida: Path):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


def main():
    parser = argparse.ArgumentParser(
        description="Mueve prefijo de tipo de vía desde 'calle' a 'tipo_via' sin canonizar y exporta a CSV."
    )
    parser.add_argument("--tabla", default="domicilios", help="Tabla origen (default: domicilios)")
    parser.add_argument("--schema", default=None, help="Esquema de la tabla (ej. public)")
    parser.add_argument("--limit", type=int, default=0, help="Filas a leer (0 = todas)")
    parser.add_argument("--preview", type=int, default=0, help="Imprime N filas antes de exportar (0 = no imprime)")
    parser.add_argument("--salida", default="", help="Nombre del CSV (si no se da, se genera con timestamp)")
    parser.add_argument("--outdir", default="", help="Carpeta destino; si no se da, usa OUTPUT_DIR o '.'")
    parser.add_argument("--env", default="", help="Ruta a archivo .env (default: autodetect)")
    parser.add_argument("--debug-sql", action="store_true", help="Imprime la consulta SQL generada")
    args = parser.parse_args()

    # Cargar .env
    if args.env:
        ok = load_dotenv(args.env, override=False)
        print(f".env cargado desde: {args.env}" if ok else f"No se pudo cargar .env desde: {args.env}")
    else:
        path = find_dotenv()
        load_dotenv(path, override=False)
        print(f".env detectado y cargado: {path}" if path else "No se encontró .env; se usarán variables del sistema.")

    # Definir outdir
    outdir_env = os.getenv("OUTPUT_DIR", "").strip()
    outdir = Path(args.outdir or outdir_env or ".")
    outdir.mkdir(parents=True, exist_ok=True)

    # Definir nombre de salida
    if args.salida.strip():
        salida_path = outdir / args.salida.strip()
    else:
        ts = datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
        salida_path = outdir / f"registros_domicilios_vdi{ts}.csv"

    conn = connect_db()
    try:
        df = leer_dataframe(
            conn=conn,
            tabla=args.tabla,
            cols=TARGET_COLS,
            limit=args.limit,
            schema=args.schema,
            debug_sql=args.debug_sql,
        )
        print(f"Registros leídos: {len(df)}")

        # Normalizar por fila
        df = df.apply(normalize_row, axis=1)

        # Vista previa
        if args.preview and args.preview > 0:
            n = min(args.preview, len(df))
            print(f"\nMuestra de {n} filas normalizadas:")
            print(df.head(n).to_string(index=False))

        # Exportar
        exportar_csv(df, salida_path)
        print(f"\nArchivo generado: {salida_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
