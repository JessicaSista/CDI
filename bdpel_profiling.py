#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profiling de datos sobre BDPel (CSV integrado D1+D2) con pandas.

Etapa 1 (primer profiling): medimos sobre los datos tal como están cargados en
BDPel, sin limpiar ni “normalizar” títulos ni etiquetas para acercarlas a otra
representación. Las únicas operaciones mínimas son las típicas al leer texto
(p. ej. separar listed_in por comas o parsear la lista de genres), y para el
catálogo de géneros se compara cada etiqueta con strip() y sin distinguir mayúsculas.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas por defecto (proyecto: carpeta raíz = padre de scripts/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BDPel = ROOT / "BDPelcsv.csv"
DEFAULT_GENRES = ROOT / "genres.txt"
DEFAULT_TOP100 = ROOT / "top100_peliculas.csv"
DEFAULT_OSCAR = ROOT / "the_oscar_award" / "the_oscar_award.csv"
DEFAULT_OUT = ROOT / "output" / "bdpel_profiling_resultados.xlsx"


# -----------------------------------------------------------------------------
# helper para no contar como "dato" las
# celdas vacías o que son solo espacios.
# -----------------------------------------------------------------------------
def texto_presente(val: Any) -> bool:
    """True si el valor es texto no vacío después de quitar espacios."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return bool(str(val).strip())


# -----------------------------------------------------------------------------
# paso parte/total y me devuelve porcentaje redondeado.
# -----------------------------------------------------------------------------
def porcentaje(parte: float, total: float) -> float:
    """Porcentaje 0–100; devuelve 0.0 si total es 0."""
    if total <= 0:
        return 0.0
    return round(100.0 * parte / total, 4)


# =============================================================================
# 1) Completitud
# =============================================================================


# -----------------------------------------------------------------------------
# medimos cuántas filas tienen descripción cargada.
# -----------------------------------------------------------------------------
def completitud_descripcion(df: pd.DataFrame, col: str = "description") -> dict[str, Any]:
    """
    Calcula el porcentaje de filas (películas/series en BDPel) que tienen
    descripción no vacía.

    Sirve para ver si se cumple el requisito de usuarios que necesitan texto
    descriptivo (y para detectar faltantes en un atributo clave del catálogo).
    """
    n = len(df)
    ok = df[col].apply(texto_presente).sum()
    return {
        "metrica": "pct_con_descripcion",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
    }


# -----------------------------------------------------------------------------
# mira si tiene año de estreno.
# -----------------------------------------------------------------------------
def completitud_anio_estreno(df: pd.DataFrame, col: str = "release_year") -> dict[str, Any]:
    """
    Porcentaje de registros con año de estreno presente (no nulo).

    """
    n = len(df)
    ok = df[col].notna().sum()
    return {
        "metrica": "pct_con_anio_estreno",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
    }


# -----------------------------------------------------------------------------
# vemos cuántos títulos tienen director no vacío.
# -----------------------------------------------------------------------------
def completitud_director(df: pd.DataFrame, col: str = "director") -> dict[str, Any]:
    """Porcentaje de registros con director no vacío (texto presente)."""
    n = len(df)
    ok = df[col].apply(texto_presente).sum()
    return {
        "metrica": "pct_con_director",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
    }


# -----------------------------------------------------------------------------
# % con columna de actores no vacía
# -----------------------------------------------------------------------------
def completitud_actores(df: pd.DataFrame, col: str = "cast_d1") -> dict[str, Any]:
    """Porcentaje de registros con actores no vacíos en la columna indicada."""
    n = len(df)
    ok = df[col].apply(texto_presente).sum()
    return {
        "metrica": "pct_con_actores",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
    }


# -----------------------------------------------------------------------------
# género puede venir como texto (listed_in) o
# como lista en string (genres). Contamos OK si en alguna de las dos hay data.
# -----------------------------------------------------------------------------
def completitud_genero(df: pd.DataFrame, cols: tuple[str, str] = ("listed_in", "genres")) -> dict[str, Any]:
    """
    Mide “hay género disponible”, no si coincide con el catálogo oficial.
    """
    n = len(df)

    # función interna: la usamos fila por fila para no repetir lógica
    def tiene_genero(row: pd.Series) -> bool:
        a, b = cols
        if texto_presente(row.get(a)):
            return True
        g = row.get(b)
        if not texto_presente(g):
            return False
        s = str(g).strip()
        if s in ("[]", "['']", ""):
            return False
        return True

    ok = df.apply(tiene_genero, axis=1).sum()
    return {
        "metrica": "pct_con_genero_informado",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
    }


# -----------------------------------------------------------------------------
# junta todas las métricas de completitud en una tabla
# -----------------------------------------------------------------------------
def tabla_completitud_resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Arma una tabla única con todas las métricas de completitud (1.x)."""
    filas = [
        completitud_descripcion(df),
        completitud_anio_estreno(df),
        completitud_director(df),
        completitud_actores(df),
        completitud_genero(df),
    ]
    return pd.DataFrame(filas)


# =============================================================================
# 2) Validez
# =============================================================================


# -----------------------------------------------------------------------------
# leemos genres.txt; guardamos cada género en minúsculas solo para poder
# comparar con cada etiqueta de BDPel en minúsculas (sin cambiar el texto original).
# -----------------------------------------------------------------------------
def cargar_generos_permitidos(ruta: Path) -> set[str]:
    """
    Lee genres.txt y devuelve el conjunto de géneros permitidos en minúsculas.
    """
    g = pd.read_csv(ruta)
    return {str(x).strip().lower() for x in g["genre"].dropna()}


def _tokens_desde_genres_d2(val: Any) -> list[str]:
    """
    Saca los strings tal como vienen en la celda `genres` (lista tipo Python en texto).
    No cambiamos el contenido salvo strip() por elemento.
    """
    if not texto_presente(val):
        return []
    s = str(val).strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if texto_presente(x)]
        except (ValueError, SyntaxError):
            pass
    return [s]


def _tokens_desde_listed_in_d1(val: Any) -> list[str]:
    """Parte listed_in por comas; cada trozo queda como en el CSV (solo strip del borde)."""
    if not texto_presente(val):
        return []
    partes = [p.strip() for p in str(val).split(",")]
    return [p for p in partes if p]


def _genero_coincide_catalogo(etiqueta: str, permitidos: set[str]) -> bool:
    """True si la etiqueta, recortada, coincide con algún género del archivo (solo casefold)."""
    t = str(etiqueta).strip()
    if not t:
        return False
    return t.lower() in permitidos


# -----------------------------------------------------------------------------
# miramos si el año es válido,
# que no sea menor a 1880 y que no sea mayor al año actual
# -----------------------------------------------------------------------------
def validez_anio_estreno(
    df: pd.DataFrame,
    col: str = "release_year",
    anio_min: int = 1880,
    anio_actual: int | None = None,
) -> dict[str, Any]:
    """
    Porcentaje de registros cuyo año de estreno es válido para el dominio:
    no nulo, mayor o igual a anio_min y no mayor al año actual (por defecto el
    año de la fecha de ejecución).

    Filas con año fuera de ese rango cuentan como inválidas para esta métrica.
    """
    if anio_actual is None:
        anio_actual = date.today().year
    n = len(df)
    y = pd.to_numeric(df[col], errors="coerce")
    ok = ((y.notna()) & (y >= anio_min) & (y <= anio_actual)).sum()
    return {
        "metrica": "pct_anio_estreno_valido",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
        "anio_min_usado": anio_min,
        "anio_max_usado": anio_actual,
    }


# -----------------------------------------------------------------------------
# % de filas con al menos un género que figure en genres.txt
# -----------------------------------------------------------------------------
def validez_generos_vs_catalogo(df: pd.DataFrame, permitidos: set[str]) -> dict[str, Any]:
    """
    Porcentaje de filas donde existe al menos un género detectado que está en
    la lista permitida 
    """
    n = len(df)
    gcol = df["genres"] if "genres" in df.columns else pd.Series([None] * n)
    lcol = df["listed_in"] if "listed_in" in df.columns else pd.Series([None] * n)

    def fila_valida(genres_val: Any, listed_val: Any) -> bool:
        tokens: list[str] = []
        tokens.extend(_tokens_desde_genres_d2(genres_val))
        for lab in _tokens_desde_listed_in_d1(listed_val):
            tokens.append(lab)
        tokens = [t for t in tokens if t]
        if not tokens:
            return False
        return any(_genero_coincide_catalogo(tok, permitidos) for tok in tokens)

    ok = sum(fila_valida(g, l) for g, l in zip(gcol, lcol))
    return {
        "metrica": "pct_con_al_menos_un_genero_valido",
        "numerador": int(ok),
        "denominador": int(n),
        "porcentaje": porcentaje(ok, n),
    }


# -----------------------------------------------------------------------------
# junta las dos métricas de validez (año y géneros) en
# una tabla para exportar al Excel en la hoja 02_validez.
# -----------------------------------------------------------------------------
def tabla_validez_resumen(df: pd.DataFrame, permitidos: set[str]) -> pd.DataFrame:
    return pd.DataFrame([validez_anio_estreno(df), validez_generos_vs_catalogo(df, permitidos)])


# =============================================================================
# 3) Consistencia y formatos
# =============================================================================


# -----------------------------------------------------------------------------
# esto muestra cómo quedó mezclado el género post integración
# -----------------------------------------------------------------------------
def consistencia_formato_generos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume cómo viene representado el género según la fila: solo texto tipo
    listed_in, solo lista D2 (genres), ambos, o ninguno.

    Ayuda a ver el efecto de la integración (dos mundos de representación).
    """
    has_li = df["listed_in"].apply(texto_presente)
    has_g = df["genres"].apply(texto_presente) & ~df["genres"].astype(str).str.strip().isin(["[]", ""])

    def bucket(li: bool, g: bool) -> str:
        if li and g:
            return "texto_y_lista"
        if li:
            return "solo_texto_netflix"
        if g:
            return "solo_lista_d2"
        return "sin_genero"

    tipo = [bucket(li, g) for li, g in zip(has_li, has_g)]
    vc = pd.Series(tipo).value_counts().rename_axis("formato_genero").reset_index(name="cantidad")
    vc["porcentaje"] = (100.0 * vc["cantidad"] / len(df)).round(4)
    return vc


# -----------------------------------------------------------------------------
# en D1 la duración era texto ("90 min", "2 Seasons") y
# en D2 viene runtime numérico. Acá clasificamos filas para ver el "desorden" de la integración.
# -----------------------------------------------------------------------------
def consistencia_formato_duracion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contrasta la duración en formato texto con la numérica `runtime`.

    Clasifica cada fila para ver coexistencia de representaciones o vacíos.
    """
    dur_txt = df["duration"].apply(texto_presente) if "duration" in df.columns else pd.Series(False, index=df.index)
    run_num = df["runtime"].notna() & (pd.to_numeric(df["runtime"], errors="coerce").notna())
    if "runtime" not in df.columns:
        run_num = pd.Series(False, index=df.index)

    def b(t: bool, r: bool) -> str:
        if t and r:
            return "texto_y_numero"
        if t:
            return "solo_texto"
        if r:
            return "solo_numero"
        return "sin_duracion"

    tipo = [b(t, r) for t, r in zip(dur_txt, run_num)]
    vc = pd.Series(tipo).value_counts().rename_axis("formato_duracion").reset_index(name="cantidad")
    vc["porcentaje"] = (100.0 * vc["cantidad"] / len(df)).round(4)
    return vc


# -----------------------------------------------------------------------------
# contamos cuántos Movie/TV Show vs MOVIE/SHOW hay.
# Sirve para mostrar que al unir datasets quedaron categorías con distinto formato.
# -----------------------------------------------------------------------------
def consistencia_valores_type(df: pd.DataFrame, col: str = "type") -> pd.DataFrame:
    """
    Frecuencia de valores en la columna tipo (Movie/TV Show vs MOVIE/SHOW, etc.).

    Permite ver heterogeneidad residual tras integrar D1 y D2.
    """
    vc = df[col].fillna("(null)").astype(str).value_counts().rename_axis("type").reset_index(name="cantidad")
    vc["porcentaje"] = (100.0 * vc["cantidad"] / len(df)).round(4)
    return vc


# =============================================================================
# 4) Duplicados
# =============================================================================


# -----------------------------------------------------------------------------
# títulos repetidos exactamente igual que en la columna title (sin unificar mayúsculas
# ni sacar puntuación). Si hay dos filas con el mismo string, sube el contador.
# -----------------------------------------------------------------------------
def duplicados_por_titulo(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    Lista valores de `title` que aparecen más de una vez (misma cadena que en BDPel).
    """
    vc = df[col].dropna().value_counts()
    dup = vc[vc > 1].reset_index()
    dup.columns = [col, "cantidad_filas"]
    return dup.sort_values("cantidad_filas", ascending=False)


# -----------------------------------------------------------------------------
# igual que la anterior pero sumamos el año 
# -----------------------------------------------------------------------------
def duplicados_por_titulo_y_anio(
    df: pd.DataFrame,
    col_title: str = "title",
    col_year: str = "release_year",
) -> pd.DataFrame:
    """
    Igual que duplicados por título pero exigiendo mismo año de estreno.
    Reduce falsos positivos (remakes, homónimos en otro año).
    """
    df2 = df[[col_title, col_year]].dropna(subset=[col_title]).copy()
    g = df2.groupby([col_title, col_year], dropna=False).size().reset_index(name="cantidad_filas")
    g = g[g["cantidad_filas"] > 1]
    return g.sort_values("cantidad_filas", ascending=False)


def resumen_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila con: grupos duplicados por título y por (título, año)."""
    d1 = duplicados_por_titulo(df)
    d2 = duplicados_por_titulo_y_anio(df)
    return pd.DataFrame(
        [
            {
                "concepto": "grupos_con_mismo_titulo_duplicado",
                "valor": int(len(d1)),
            },
            {
                "concepto": "grupos_con_mismo_titulo_y_anio_duplicado",
                "valor": int(len(d2)),
            },
            {
                "concepto": "filas_totales_en_duplicados_titulo",
                "valor": int(d1["cantidad_filas"].sum()) if len(d1) else 0,
            },
        ]
    )


# =============================================================================
# 5) Cobertura fuentes externas
# =============================================================================


# -----------------------------------------------------------------------------
# lee el CSV del top 100 (rank, title, year) para después
# comparar contra los títulos que tenemos en BDPel.
# -----------------------------------------------------------------------------
def cargar_top100(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta)


# -----------------------------------------------------------------------------
# películas ganadoras (winner=True) en el rango de años; nombres tal como vienen en la columna film.
# -----------------------------------------------------------------------------
def cargar_oscar_ganadoras(ruta: Path, anio_min: int = 1927, anio_max: int = 2024) -> set[Any]:
    """
    Conjunto de valores `film` distintos que ganaron al menos un premio (winner),
    con year_film entre anio_min y anio_max. Sin transformar el texto del título.
    """
    o = pd.read_csv(ruta)
    o = o[(o["year_film"] >= anio_min) & (o["year_film"] <= anio_max)]
    o = o[o["winner"].astype(str).str.lower().isin(("true", "1", "yes"))]
    return set(o["film"].dropna().unique())


# -----------------------------------------------------------------------------
# para cada fila del top 100 miramos si ese string de título existe igual en BDPel.title.
# -----------------------------------------------------------------------------
def cobertura_top100(df: pd.DataFrame, ruta_top100: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Para cada título del archivo top100, indica si el mismo valor aparece en
    la columna `title` de BDPel (igualdad exacta de lo que vino en cada CSV).
    """
    t = cargar_top100(ruta_top100)
    catalog = set(df["title"].dropna().unique())
    filas = []
    for _, row in t.iterrows():
        tit = row.get("title")
        en = False if pd.isna(tit) else tit in catalog
        filas.append(
            {
                "rank": row.get("rank"),
                "title_top100": tit,
                "year_top100": row.get("year"),
                "en_bdpel": bool(en),
            }
        )
    det = pd.DataFrame(filas)
    n = len(det)
    ok = int(det["en_bdpel"].sum())
    resumen = pd.DataFrame(
        [
            {
                "metrica": "pct_top100_presente_en_bdpel",
                "numerador": ok,
                "denominador": n,
                "porcentaje": porcentaje(ok, n),
            }
        ]
    )
    return det, resumen


# -----------------------------------------------------------------------------
# miro cuántas películas ganadoras (set del archivo Oscar)
# están presentes en BDPel por título. Es aproximado: no cruzamos año de estreno.
# -----------------------------------------------------------------------------
def cobertura_oscar(df: pd.DataFrame, ruta_oscar: Path) -> pd.DataFrame:
    """
    Intersección entre títulos ganadores (columna film, valores únicos) y
    títulos presentes en BDPel (`title`), comparando el string tal como está en cada tabla.
    """
    oscar_titles = cargar_oscar_ganadoras(ruta_oscar)
    catalog = set(df["title"].dropna().unique())
    presentes = oscar_titles & catalog
    n = len(oscar_titles)
    ok = len(presentes)
    return pd.DataFrame(
        [
            {
                "metrica": "pct_oscar_ganadoras_presentes_en_bdpel",
                "numerador": ok,
                "denominador": max(n, 0),
                "porcentaje": porcentaje(ok, n) if n else 0.0,
                "nota": "Match exacto title==film; set unico film ganador; sin cruzar anio",
            }
        ]
    )


# =============================================================================
# 6) Distribución
# =============================================================================


# -----------------------------------------------------------------------------
# cuenta cuántas películas/series hay por año de estreno.
# Sirve para histogramas y ver si el dataset está viejo o tiene mucho contenido nuevo.
# -----------------------------------------------------------------------------
def distribucion_por_anio(df: pd.DataFrame, col: str = "release_year") -> pd.DataFrame:
    """
    Cantidad de registros por año de estreno (histograma simple).
    """
    y = pd.to_numeric(df[col], errors="coerce")
    vc = y.dropna().astype(int).value_counts().sort_index().rename_axis("release_year").reset_index(name="cantidad")
    return vc


# -----------------------------------------------------------------------------
# saca qué % cayó en los últimos N años (por defecto 10)
# mirando el año actual del sistema. 
# -----------------------------------------------------------------------------
def porcentaje_recientes_ultimos_anios(df: pd.DataFrame, n_anios: int = 10, col: str = "release_year") -> dict[str, Any]:
    """
    Porcentaje de registros cuyo año de estreno cae en los últimos `n_anios`
    respecto al año de referencia (por defecto el año actual).

    Ejemplo con n_anios=10 y año 2026: intervalo (2016 .. 2026] en años enteros.
    """
    ref = date.today().year
    y = pd.to_numeric(df[col], errors="coerce")
    min_y = ref - n_anios + 1
    ok = ((y.notna()) & (y >= min_y) & (y <= ref)).sum()
    total = len(df)
    return {
        "metrica": f"pct_ultimos_{n_anios}_anios",
        "anio_referencia": ref,
        "rango_inclusive": f"{min_y}-{ref}",
        "numerador": int(ok),
        "denominador": int(total),
        "porcentaje": porcentaje(ok, total),
    }


def _explode_generos_fila(row: pd.Series) -> list[str]:
    return _explode_generos_columnas(row.get("genres"), row.get("listed_in"))


# -----------------------------------------------------------------------------
# lista todas las etiquetas de género de una fila como están en el CSV (sin mapear).
# -----------------------------------------------------------------------------
def _explode_generos_columnas(genres_val: Any, listed_in_val: Any) -> list[str]:
    """Junta tokens desde `genres` y trozos desde `listed_in`, sin reescribirlos."""
    out: list[str] = []
    for t in _tokens_desde_genres_d2(genres_val):
        if t:
            out.append(t)
    for lab in _tokens_desde_listed_in_d1(listed_in_val):
        out.append(lab)
    return [x for x in out if x]


# -----------------------------------------------------------------------------
# ranking de etiquetas de género tal como aparecen en BDPel (texto + lista), sin unificar sinónimos.
# El % es sobre el total de etiquetas contadas, no sobre cantidad de filas.
# -----------------------------------------------------------------------------
def distribucion_por_genero(df: pd.DataFrame, top: int = 40) -> pd.DataFrame:
    """
    Frecuencia de etiquetas de género tal como están en `genres` y `listed_in`
    (sin normalizar a un vocabulario común).
    """
    todos: list[str] = []
    gcol = df["genres"] if "genres" in df.columns else pd.Series([None] * len(df))
    lcol = df["listed_in"] if "listed_in" in df.columns else pd.Series([None] * len(df))
    for gv, lv in zip(gcol, lcol):
        todos.extend(_explode_generos_columnas(gv, lv))
    if not todos:
        return pd.DataFrame(columns=["etiqueta_como_en_bdpel", "cantidad", "porcentaje"])
    vc = pd.Series(todos).value_counts().head(top).rename_axis("etiqueta_como_en_bdpel").reset_index(name="cantidad")
    vc["porcentaje"] = (100.0 * vc["cantidad"] / len(todos)).round(4)
    return vc


# =============================================================================
# Export Excel + main
# =============================================================================


# -----------------------------------------------------------------------------
# esta función corre todas las métricas y escribe el
# Excel con una hoja por tema (así después graficamos en otro lado).
# -----------------------------------------------------------------------------
def exportar_excel(
    df: pd.DataFrame,
    permitidos: set[str],
    ruta_top100: Path,
    ruta_oscar: Path,
    salida: Path,
) -> None:
    salida.parent.mkdir(parents=True, exist_ok=True)
    comp = tabla_completitud_resumen(df)
    val = tabla_validez_resumen(df, permitidos)
    top_det, top_res = cobertura_top100(df, ruta_top100)
    osc = cobertura_oscar(df, ruta_oscar)
    dup_tit = duplicados_por_titulo(df)
    dup_ta = duplicados_por_titulo_y_anio(df)
    dup_sum = resumen_duplicados(df)

    hoja_resumen = pd.concat(
        [
            comp.assign(bloque="completitud"),
            val.assign(bloque="validez"),
        ],
        ignore_index=True,
    )
    rec = pd.DataFrame([porcentaje_recientes_ultimos_anios(df)])
    dist_a = distribucion_por_anio(df)
    dist_g = distribucion_por_genero(df)

    with pd.ExcelWriter(salida, engine="openpyxl") as w:
        hoja_resumen.to_excel(w, sheet_name="Resumen_KPIs", index=False)
        comp.to_excel(w, sheet_name="01_completitud", index=False)
        val.to_excel(w, sheet_name="02_validez", index=False)
        consistencia_formato_generos(df).to_excel(w, sheet_name="03a_formato_genero", index=False)
        consistencia_formato_duracion(df).to_excel(w, sheet_name="03b_formato_duracion", index=False)
        consistencia_valores_type(df).to_excel(w, sheet_name="03c_valores_type", index=False)
        dup_sum.to_excel(w, sheet_name="04_resumen_duplicados", index=False)
        dup_tit.head(500).to_excel(w, sheet_name="04b_dup_titulo", index=False)
        dup_ta.head(500).to_excel(w, sheet_name="04c_dup_titulo_anio", index=False)
        top_det.to_excel(w, sheet_name="05_top100_detalle", index=False)
        top_res.to_excel(w, sheet_name="05_top100_resumen", index=False)
        osc.to_excel(w, sheet_name="06_oscar_resumen", index=False)
        dist_a.to_excel(w, sheet_name="07_distribucion_anio", index=False)
        rec.to_excel(w, sheet_name="08_recientes_10_anios", index=False)
        dist_g.to_excel(w, sheet_name="09_distribucion_generos", index=False)


# -----------------------------------------------------------------------------
# entrypoint: lee el CSV de BDPel, carga géneros permitidos
# y genera el Excel por defecto (o rutas que pases por consola).
# -----------------------------------------------------------------------------
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Profiling BDPel -> Excel")
    p.add_argument("--bdpel", type=Path, default=DEFAULT_BDPel)
    p.add_argument("--genres", type=Path, default=DEFAULT_GENRES)
    p.add_argument("--top100", type=Path, default=DEFAULT_TOP100)
    p.add_argument("--oscar", type=Path, default=DEFAULT_OSCAR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    df = pd.read_csv(args.bdpel, low_memory=False)
    permitidos = cargar_generos_permitidos(args.genres)
    exportar_excel(df, permitidos, args.top100, args.oscar, args.out)
    print(f"Listo: {args.out} ({len(df)} filas)")


if __name__ == "__main__":
    main()
