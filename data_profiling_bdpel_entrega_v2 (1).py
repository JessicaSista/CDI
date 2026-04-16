# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "salida_profiling_entrega_v2"

# Años plausibles para cine/TV (ajustable al contexto del curso)
ANIO_MIN = 1880
ANIO_MAX = 2026
ANIO_REFERENCIA = 2026
VENTANA_RECIENTE_ANIOS = 10

CSV_OSCAR = BASE / "the_oscar_award" / "the_oscar_award.csv"
TXT_GENEROS_VALIDOS = BASE / "genres.txt"
TXT_TOP100 = BASE / "top100_peliculas.txt"

# Nombres de columnas en el export actual de BDPel
COL_ID = "idpel"
COL_TITULO = "titulo"
COL_DESC = "descripcion"
COL_DIR = "directores"
COL_ACT = "actores"
COL_GEN = "generos"
COL_DUR = "duracion"
COL_ANO = "anoestreno"


def resolver_bdpel_csv() -> Path:
    """
    Ubica el CSV de la base integrada.
    """
    for cand in (BASE / "BDPel.csv", BASE / "bdpel.csv"):
        if cand.is_file():
            return cand
    return BASE / "BDPel.csv"


def cargar_bdpel(path: Path) -> pd.DataFrame:
    """
    Lee BDPel en un DataFrame.

    """
    if not path.is_file():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    # Unificamos nombres a minúsculas por si el export trae mayúsculas distintas
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def es_vacio(col: pd.Series) -> pd.Series:
    """
    Marca celdas que tratamos como “sin dato” para el conteo de faltantes.

    Para texto: cuenta NaN, strings vacíos después de strip, y esos “nan”/“None” que
    a veces vienen como texto porque algo en la integración los serializó mal.
    Para números: solo NaN (si idpel/ano vienen como int, no hay “vacío” intermedio).
    """
    if col.dtype == object:
        return col.isna() | (col.astype(str).str.strip().isin(["", "nan", "None"]))
    return col.isna()


def _norm_titulo(s: Any) -> str:
    """Normaliza título para cruzar con Oscar: minúsculas y sin espacios al borde."""
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def _titulo_clave_matching(s: Any) -> str:
    """
    Clave para comparar títulos entre BDPel y listas externas.

    Pasamos a minúsculas, unificamos guiones largos/cortos y sacamos el año entre
    paréntesis al final si viene (p. ej. \"The Godfather (1972)\" → misma clave que en la lista sin año).
    """
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("–", "-")
    t = re.sub(r"\s*[\(（]\s*\d{4}\s*[\)）]\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def leer_titulos_top100(path: Path) -> list[str]:
    """
    Lee top100_peliculas.txt: una película por línea = una posición del ranking.

    Pueden repetirse títulos (el ranking original tiene duplicados entre tramos);
    el porcentaje de cobertura cuenta cuántas de esas posiciones tienen match en BDPel.
    """
    if not path.is_file():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def calcular_metricas_referencias_y_recencia(df: pd.DataFrame) -> dict[str, Any]:
    """
    Métricas pedidas para el informe: Top 100, Oscar (ganadoras), recencia (últimos 10 años).

    Los porcentajes de cobertura son \"títulos del referencial presentes en BDPel\" / |referencial|.
    La recencia es % de filas con año en [referencia−10, referencia] y dentro del rango de dominio.
    """
    n = len(df)
    out: dict[str, Any] = {"n_filas_bdpel": n}

    # --- Recencia (año de estreno en los últimos VENTANA_RECIENTE_ANIOS años) ---
    if COL_ANO in df.columns:
        y = pd.to_numeric(df[COL_ANO], errors="coerce")
        umbral = ANIO_REFERENCIA - VENTANA_RECIENTE_ANIOS
        en_ventana = (y >= umbral) & (y <= ANIO_MAX)
        valido_dom = (y >= ANIO_MIN) & (y <= ANIO_MAX)
        out["ano_umbral_reciente"] = umbral
        out["filas_ano_en_ultimos_10_anios"] = int(en_ventana.sum())
        out["pct_filas_ano_ultimos_10"] = 100.0 * float(en_ventana.sum()) / max(n, 1)
        nv = int(valido_dom.sum())
        out["filas_con_ano_valido_en_dominio"] = nv
        if nv > 0:
            out["pct_sobre_solo_ano_valido_ultimos_10"] = (
                100.0 * float((en_ventana & valido_dom).sum()) / nv
            )
        else:
            out["pct_sobre_solo_ano_valido_ultimos_10"] = None

    if COL_TITULO not in df.columns:
        return out

    claves_bdp = set(df[COL_TITULO].dropna().map(_titulo_clave_matching))
    claves_bdp.discard("")

    # --- Top 100: % = posiciones del ranking presentes en BDPel / nº de líneas del archivo ---
    titulos_ref = leer_titulos_top100(TXT_TOP100)
    n_ref = len(titulos_ref)
    posiciones_en_bdpel = sum(
        1 for t in titulos_ref if t and _titulo_clave_matching(t) in claves_bdp
    )
    out["top100_entradas_en_lista"] = n_ref
    out["top100_posiciones_encontradas_en_bdpel"] = posiciones_en_bdpel
    out["pct_top100_en_bdpel"] = 100.0 * posiciones_en_bdpel / max(n_ref, 1)
    # Auxiliar: títulos únicos del archivo vs cuántos de esos únicos aparecen en BDPel
    claves_top = {_titulo_clave_matching(t) for t in titulos_ref}
    claves_top.discard("")
    top_en_bdp = claves_top & claves_bdp
    out["top100_titulos_distintos_en_lista"] = len(claves_top)
    out["top100_titulos_distintos_presentes_en_bdpel"] = len(top_en_bdp)

    # --- Oscar: películas ganadoras distintas (1927–2024) ---
    if CSV_OSCAR.is_file():
        oscar = pd.read_csv(CSV_OSCAR, encoding="utf-8")
        if {"winner", "year_film", "film"}.issubset(oscar.columns):
            wcol = oscar["winner"]
            if wcol.dtype == object:
                ganador = wcol.astype(str).str.lower().isin(["true", "1", "yes"])
            else:
                ganador = wcol == True  # noqa: E712
            win = oscar.loc[ganador].copy()
            win = win[(win["year_film"] >= 1927) & (win["year_film"] <= 2024)]
            films = set(win["film"].dropna().map(_titulo_clave_matching))
            films.discard("")
            out["oscar_peliculas_ganadoras_distintas"] = len(films)
            inter_o = films & claves_bdp
            out["oscar_coincidencias_en_bdpel"] = len(inter_o)
            out["pct_oscar_ganadoras_en_bdpel"] = 100.0 * len(inter_o) / max(len(films), 1)
        else:
            out["oscar_error"] = "columnas esperadas ausentes"
    else:
        out["oscar_error"] = "archivo no encontrado"

    # --- Géneros válidos (genres.txt): % de filas con ≥1 término de la taxonomía ---
    gv = _leer_generos_validos(TXT_GENEROS_VALIDOS)
    if COL_GEN in df.columns and gv:
        mgv = calcular_metricas_generos_lista_valida(df, gv)
        out["genres_txt_cantidad_terminos_lista"] = mgv.get("generos_validos_en_txt")
        out["genres_txt_filas_con_alguno_valido"] = mgv.get("filas_con_al_menos_un_genero_valido")
        out["genres_txt_pct_filas_con_alguno_valido"] = mgv.get("pct_sobre_total_filas")
        out["genres_txt_pct_sobre_filas_con_genero_no_vacio"] = mgv.get(
            "pct_sobre_filas_con_genero_no_vacio"
        )
        out["genres_txt_filas_genero_no_vacio_sin_match"] = mgv.get(
            "filas_con_genero_pero_ningun_valido"
        )
    elif COL_GEN in df.columns:
        out["genres_txt_error"] = "no se pudo leer la lista en genres.txt"

    return out


def formato_bloque_referencias_recencia(m: dict[str, Any]) -> str:
    """Texto listo para pegar en ST2 / informe."""
    lineas = [
        "\n--- Referencias externas y recencia ---",
        f"Ventana 'reciente': año de estreno >= {m.get('ano_umbral_reciente', '?')} "
        f"(últimos {VENTANA_RECIENTE_ANIOS} años respecto de {ANIO_REFERENCIA}).",
    ]
    if "pct_filas_ano_ultimos_10" in m:
        lineas.append(
            f"% de filas con año en esa ventana (y <= {ANIO_MAX}): "
            f"{m['pct_filas_ano_ultimos_10']:.2f}% "
            f"({m.get('filas_ano_en_ultimos_10_anios', 0)} / {m.get('n_filas_bdpel', 0)} filas)."
        )
        if m.get("pct_sobre_solo_ano_valido_ultimos_10") is not None:
            lineas.append(
                f"  (Sobre filas con año válido en [{ANIO_MIN},{ANIO_MAX}]: "
                f"{m['pct_sobre_solo_ano_valido_ultimos_10']:.2f}%.)"
            )
    lineas.append("")
    if m.get("genres_txt_pct_filas_con_alguno_valido") is not None:
        lineas.append(
            f"Géneros válidos ({TXT_GENEROS_VALIDOS.name}): "
            f"{m.get('genres_txt_filas_con_alguno_valido', 0)} filas con al menos un género de la lista "
            f"→ {float(m.get('genres_txt_pct_filas_con_alguno_valido') or 0):.2f}% del total de filas."
        )
        if m.get("genres_txt_pct_sobre_filas_con_genero_no_vacio") is not None:
            lineas.append(
                f"  Sobre filas con género no vacío: "
                f"{float(m['genres_txt_pct_sobre_filas_con_genero_no_vacio']):.2f}% "
                f"({m.get('genres_txt_filas_genero_no_vacio_sin_match', 0)} filas sin ningún término reconocible)."
            )
    elif m.get("genres_txt_error"):
        lineas.append(f"Géneros válidos: {m['genres_txt_error']}")
    lineas.append("")
    lineas.append(
        f"Top 100 ({TXT_TOP100.name}): "
        f"{m.get('top100_posiciones_encontradas_en_bdpel', 0)} de "
        f"{m.get('top100_entradas_en_lista', 0)} posiciones del ranking tienen título equivalente en BDPel "
        f"→ {float(m.get('pct_top100_en_bdpel') or 0):.2f}%."
    )
    if m.get("top100_titulos_distintos_en_lista") is not None:
        lineas.append(
            f"  (Aux.: {m.get('top100_titulos_distintos_presentes_en_bdpel', 0)} / "
            f"{m.get('top100_titulos_distintos_en_lista', 0)} títulos únicos del listado presentes en BDPel.)"
        )
    lineas.append(
        f"Oscar (ganadoras 1927–2024 en CSV; películas distintas): "
        f"{m.get('oscar_coincidencias_en_bdpel', 0)} / {m.get('oscar_peliculas_ganadoras_distintas', 0)} "
        f"→ {float(m.get('pct_oscar_ganadoras_en_bdpel') or 0):.2f}%."
    )
    if m.get("oscar_error"):
        lineas.append(f"  Nota Oscar: {m['oscar_error']}")
    lineas.append(
        "Criterio de matching: clave normalizada (minúsculas, guiones unificados, sin (año) final); "
        "sin fuzzy matching ni resolución de homónimos."
    )
    return "\n".join(lineas)


def _leer_generos_validos(path: Path) -> set[str]:
    """
    Arma el conjunto de géneros “válidos” desde genres.txt (tercera columna).

    """
    if not path.is_file():
        return set()
    valid: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[2]:
            valid.add(parts[2].strip().lower())
    return valid


def _parse_generos_lista_python(raw: Any) -> list[str]:
    """
    Intenta leer géneros cuando vienen como lista tipo "['drama', 'thriller']".

    """
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        items = [x.strip().strip("'").strip('"') for x in inner.split(",")]
        return [x for x in items if x]
    return [s]


def _quitar_sufijos_netflix_genero(p: str) -> str:
    """Quita sufijos típicos de etiquetas estilo Netflix en minúsculas."""
    p = p.strip()
    while p:
        hit = False
        for suf in (
            " movies",
            " tv shows",
            " tv show",
            " shows",
            " series",
            " films",
            " movie",
            " documentaries",
            " documentary",
        ):
            if p.endswith(suf):
                p = p[: -len(suf)].strip()
                hit = True
                break
        if not hit:
            break
    return p


def _texto_tiene_subcadena_en_validos(texto: str, valid: set[str]) -> bool:
    """
    Comprueba si alguna subsecuencia contigua de palabras (hasta 4) coincide con un género válido.

    Sirve para \"science fiction\", \"romantic comedy\", etc.
    """
    texto = texto.strip().lower()
    if not texto:
        return False
    if texto in valid:
        return True
    if len(texto) > 3 and texto.endswith("s") and not texto.endswith("ss"):
        if texto[:-1] in valid:
            return True
    words = texto.split()
    n = len(words)
    for i in range(n):
        for j in range(i + 1, min(i + 5, n) + 1):
            sub = " ".join(words[i:j])
            if sub in valid:
                return True
            if len(sub) > 3 and sub.endswith("s") and not sub.endswith("ss") and sub[:-1] in valid:
                return True
    return False


def fila_tiene_algun_genero_valido(raw: Any, valid: set[str]) -> bool:
    """
    True si en la celda de géneros aparece al menos un término reconocido en genres.txt.

    - Formato lista D2: ítems en minúsculas comparados al set.
    - Texto con comas (Netflix): segmentos, & /, sufijos \"... Movies\", y subcadenas válidas.
    """
    if not valid or genero_esta_vacio(raw):
        return False
    s = str(raw).strip()
    if s.startswith("["):
        for it in _parse_generos_lista_python(raw):
            t = it.strip().lower()
            if t in valid:
                return True
            if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and t[:-1] in valid:
                return True
        return False
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        for piece in re.split(r"\s*&\s*|\s*/\s*", part):
            p = _quitar_sufijos_netflix_genero(piece.lower())
            if _texto_tiene_subcadena_en_validos(p, valid):
                return True
    return False


def calcular_metricas_generos_lista_valida(df: pd.DataFrame, valid: set[str]) -> dict[str, Any]:
    """Conteos para el informe: filas con ≥1 género en genres.txt."""
    n = len(df)
    out: dict[str, Any] = {"generos_validos_en_txt": len(valid)}
    if COL_GEN not in df.columns or not valid:
        out["error"] = "sin columna generos o lista vacía"
        return out
    con_dato = ~df[COL_GEN].map(genero_esta_vacio)
    n_con_dato = int(con_dato.sum())
    ok = df[COL_GEN].map(lambda x: fila_tiene_algun_genero_valido(x, valid))
    n_ok = int(ok.sum())
    out["filas_con_al_menos_un_genero_valido"] = n_ok
    out["pct_sobre_total_filas"] = 100.0 * n_ok / max(n, 1)
    out["filas_con_genero_no_vacio"] = n_con_dato
    if n_con_dato > 0:
        out["pct_sobre_filas_con_genero_no_vacio"] = 100.0 * int((ok & con_dato).sum()) / n_con_dato
    else:
        out["pct_sobre_filas_con_genero_no_vacio"] = None
    out["filas_con_genero_pero_ningun_valido"] = int((con_dato & ~ok).sum())
    return out


def genero_esta_vacio(raw: Any) -> bool:
    """
    Dice si la columna géneros quedó sin información útil.

    Tratamos como vacío: NULL, "", "[]", lista parseada sin elementos, etc.
    """
    if pd.isna(raw):
        return True
    s = str(raw).strip()
    if s.lower() in ("", "nan", "none"):
        return True
    if s in ("[]", "['']", '[""]'):
        return True
    items = _parse_generos_lista_python(raw)
    return len(items) == 0


def st2_profiling(df: pd.DataFrame) -> str:
    """
    Arma el bloque principal de ST2 (data profiling “clásico”).

    Incluye volumen, tipos que ve pandas, % de faltantes por columna, duplicados de fila
    y de título, y un par de patrones sobre año y duración que nos sirven después para
    hablar de consistencia en el informe.
    """
    n = len(df)
    out: list[str] = []
    out.append("=== ST2 - DATA PROFILING - BDPel ===\n")
    out.append(f"Filas: {n}")
    out.append(f"Columnas: {len(df.columns)}\n")

    out.append("--- Dtypes ---")
    out.extend([f"{c}: {df[c].dtype}" for c in df.columns])

    out.append("\n--- Faltantes ---")
    for c in df.columns:
        m = int(es_vacio(df[c]).sum())
        out.append(f"{c}: {m} ({100 * m / max(n, 1):.2f}%)")

    out.append("\n--- Duplicados ---")
    out.append(f"Filas duplicadas exactas: {int(df.duplicated().sum())}")
    if COL_TITULO in df.columns:
        out.append(f"titulo duplicado (mismo string): {int(df[COL_TITULO].duplicated().sum())}")

    out.append("\n--- Patrones (año) ---")
    if COL_ANO in df.columns:
        y = pd.to_numeric(df[COL_ANO], errors="coerce")
        fuera = int(((y < ANIO_MIN) | (y > ANIO_MAX)).sum())
        out.append(f"anoestreno fuera de [{ANIO_MIN},{ANIO_MAX}]: {fuera}")
        out.append(f"anoestreno min / max (numérico): {y.min()} / {y.max()}")

    out.append("\n--- Patrones (duración, heurística) ---")
    if COL_DUR in df.columns:
        d = df[COL_DUR].dropna().astype(str).str.strip()
        con_min = d.str.contains("min", case=False, na=False).sum()
        con_season = d.str.contains("season", case=False, na=False).sum()
        solo_digitos = d.str.match(r"^\d+$", na=False).sum()
        out.append(f"duracion con 'min': {con_min} / {len(d)} no nulos")
        out.append(f"duracion con 'season': {con_season} / {len(d)} no nulos")
        out.append(f"duracion solo número (ej. runtime D2): {solo_digitos} / {len(d)} no nulos")

    out.append("\n--- Patrones (géneros) ---")
    if COL_GEN in df.columns:
        g = df[COL_GEN].dropna().astype(str).str.strip()
        parece_lista = g.str.startswith("[").sum()
        out.append(f"generos que empiezan con '[' (estilo lista): {parece_lista} / {len(g)} no nulos")
        vac_sem = int(df[COL_GEN].map(genero_esta_vacio).sum())
        out.append(
            f"generos sin informacion util ([] o sin items, etc.): {vac_sem} "
            f"(puede ser >0 aunque 'faltantes' de NaN sea bajo)"
        )

    _mref = calcular_metricas_referencias_y_recencia(df)
    out.append(formato_bloque_referencias_recencia(_mref))

    out.append("")
    return "\n".join(out)


def br_resumen(df: pd.DataFrame) -> str:
    """
    Resumen corto de reglas de negocio que podemos medir con este CSV.

    BR1: no integrar filas sin título → esperamos 0 sin titulo.
    BR4: en el modelo de contexto piden varios campos obligatorios; acá contamos cuántas
    filas les falta cada uno (no “arregla” nada, solo deja números para el informe).
    """
    n = len(df)
    out: list[str] = ["=== BR (resumen) - BDPel ===", ""]
    if COL_TITULO in df.columns:
        out.append(f"BR1 sin titulo: {int(es_vacio(df[COL_TITULO]).sum())}/{n}")
    if COL_ANO in df.columns:
        y = pd.to_numeric(df[COL_ANO], errors="coerce")
        out.append(f"BR4 anoestreno nulo o no numérico: {int(y.isna().sum())}/{n}")
        out.append(f"BR4 anoestreno fuera de rango [{ANIO_MIN},{ANIO_MAX}]: {int(((y < ANIO_MIN) | (y > ANIO_MAX)).sum())}/{n}")
    if COL_DIR in df.columns:
        out.append(f"BR4 sin director(es): {int(es_vacio(df[COL_DIR]).sum())}/{n}")
    if COL_ACT in df.columns:
        out.append(f"BR4 sin actores: {int(es_vacio(df[COL_ACT]).sum())}/{n}")
    if COL_GEN in df.columns:
        vac_gen = df[COL_GEN].map(lambda x: genero_esta_vacio(x))
        out.append(f"BR4 sin género útil: {int(vac_gen.sum())}/{n}")
    out.append("")
    return "\n".join(out)


def analizar_completitud(df: pd.DataFrame) -> str:
    """
    Dimensión completitud: qué tan “llena” viene cada columna.

    Es básicamente el % de faltantes por atributo; nos ayuda a conectar con los problemas
    P1 del informe (huecos en director, actores, etc.).
    """
    lineas = ["=== COMPLETITUD — BDPel ===", f"Filas: {len(df)}"]
    for col in df.columns:
        vac = es_vacio(df[col]).sum()
        pct = 100.0 * vac / len(df) if len(df) else 0.0
        lineas.append(f"  {col}: vacíos {vac} ({pct:.1f}%)")
    return "\n".join(lineas)


def analizar_unicidad(df: pd.DataFrame) -> str:
    """
    Dimensión unicidad: filas repetidas y títulos repetidos.

    Si hay muchos títulos duplicados puede ser la misma obra con distinto formato (P3),
    o ruido de integración; el profiling no decide cuál es, solo cuenta.
    """
    lineas = ["=== UNICIDAD — BDPel ==="]
    lineas.append(f"Filas duplicadas (todas las columnas): {int(df.duplicated().sum())}")
    if COL_TITULO in df.columns:
        lineas.append(f"Títulos únicos (nunique): {df[COL_TITULO].nunique(dropna=True)}")
        lineas.append(f"Filas con titulo repetido: {int(df[COL_TITULO].duplicated().sum())}")
    return "\n".join(lineas)


def analizar_exactitud_sintactica(df: pd.DataFrame) -> str:
    """
    Exactitud sintáctica (liviana): años fuera de un rango razonable.

    No valida contra el mundo real fila por fila; solo detecta valores imposibles o
    sospechosos tipo 2038 si aparecieran.
    """
    lineas = [f"=== EXACTITUD (rango año) — BDPel ==="]
    if COL_ANO not in df.columns:
        return "\n".join(lineas + ["Sin columna anoestreno."])
    y = pd.to_numeric(df[COL_ANO], errors="coerce")
    fuera = (y < ANIO_MIN) | (y > ANIO_MAX)
    n = int(fuera.sum())
    lineas.append(f"anoestreno fuera de [{ANIO_MIN}, {ANIO_MAX}]: {n} filas")
    if n > 0 and COL_TITULO in df.columns:
        muestra = df.loc[fuera, [COL_TITULO, COL_ANO]].head(25)
        lineas.append("Muestra (hasta 25):")
        lineas.append(muestra.to_string(index=False))
    return "\n".join(lineas)


def analizar_frescura(df: pd.DataFrame) -> str:
    """
    Frescura aproximada: proporción de títulos “recientes” según el año de estreno.

    Usamos la ventana de 10 años del contexto (>= 2016 con referencia 2026). Es un proxy:
    no sabemos fecha de última actualización del catálogo en todas las filas.
    """
    lineas = ["=== FRESCURA (proxy por año) — BDPel ==="]
    if COL_ANO not in df.columns:
        return "\n".join(lineas + ["Sin anoestreno."])
    y = pd.to_numeric(df[COL_ANO], errors="coerce")
    umbral = ANIO_REFERENCIA - VENTANA_RECIENTE_ANIOS
    reciente = y >= umbral
    valid = y.notna()
    lineas.append(
        f"Recientes (anoestreno >= {umbral}): {int(reciente.sum())} filas "
        f"({100 * reciente.mean():.1f}% sobre filas con año parseable)"
    )
    lineas.append(f"Min / max año: {y.min()} / {y.max()}")
    return "\n".join(lineas)


def profiling_chequeos_extendidos(df: pd.DataFrame) -> str:
    """
    Cosas extra que no entran en un describe() pedorro pero suman para ST2.

    Cardinalidad por columna, estadísticos de lo numérico, longitudes de texto en campos
    cortos, y señales de mezcla D1/D2 en duración/género (formatos distintos conviviendo).
    """
    lineas: list[str] = ["=== Chequeos extendidos — BDPel ===", "", "--- Cardinalidad (nunique) ---"]
    for col in df.columns:
        try:
            nu = df[col].nunique(dropna=True)
        except TypeError:
            nu = df[col].nunique()
        lineas.append(f"  {col}: {nu} distintos")

    lineas.extend(["", "--- describe() columnas numéricas ---"])
    num = df.select_dtypes(include=["number"])
    if num.shape[1]:
        lineas.append(num.describe().to_string())
    else:
        lineas.append("(No hay columnas numéricas detectadas.)")

    lineas.extend(["", "--- Longitud de texto (object), sin descripcion ---"])
    for col in df.columns:
        if df[col].dtype != object or col == COL_DESC:
            continue
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        lens = s.str.len()
        lineas.append(f"  {col}: media={lens.mean():.1f}, max={lens.max()}, min={lens.min()}")

    lineas.append("")
    return "\n".join(lineas)


def verificar_reglas_negocio_extendido(df: pd.DataFrame) -> str:
    """
    Más reglas de negocio que en el resumen corto: BR2/BR3/BR5 donde aplica.

    BR2: no homogeneizamos en la carga → miramos convivencia de formatos.
    BR3: cobertura contra ganadoras Oscar (match simple por título).
    BR5: items de género en formato lista vs genres.txt (cota imperfecta).
    """
    n = len(df)
    lineas: list[str] = ["", "=== REGLAS DE NEGOCIO (BR1–BR5) — detalle ===", ""]

    if COL_TITULO in df.columns:
        sin_t = int(es_vacio(df[COL_TITULO]).sum())
        lineas.append(f"BR1 — filas sin titulo: {sin_t} / {n} (esperable 0)")

    lineas.append("")
    lineas.append("BR2 — formatos mezclados (esperable en integración mínima)")
    if COL_DUR in df.columns:
        d = df[COL_DUR].dropna().astype(str)
        lineas.append(f"  duracion: {d.nunique()} valores distintos (muchos formatos)")
    if COL_GEN in df.columns:
        g = df[COL_GEN].dropna().astype(str)
        lista_py = g.str.strip().str.startswith("[").sum()
        lineas.append(f"  generos con '[' (tipo lista): {lista_py} / {len(g)} no nulos")

    lineas.append("")
    lineas.append("BR3 — Cobertura referenciales (Top 100 + Oscar 1927–2024)")
    _mr = calcular_metricas_referencias_y_recencia(df)
    lineas.append(
        f"  Top 100 (posiciones en BDPel / entradas en lista): "
        f"{_mr.get('top100_posiciones_encontradas_en_bdpel', 0)} / "
        f"{_mr.get('top100_entradas_en_lista', 0)} → "
        f"{float(_mr.get('pct_top100_en_bdpel') or 0):.2f}%"
    )
    if _mr.get("oscar_peliculas_ganadoras_distintas") is not None:
        lineas.append(
            f"  Oscar (ganadoras distintas): {_mr.get('oscar_coincidencias_en_bdpel', 0)} / "
            f"{_mr.get('oscar_peliculas_ganadoras_distintas', 0)} → "
            f"{float(_mr.get('pct_oscar_ganadoras_en_bdpel') or 0):.2f}%"
        )
    elif _mr.get("oscar_error"):
        lineas.append(f"  Oscar: {_mr['oscar_error']}")
    lineas.append("  Matching: clave normalizada (_titulo_clave_matching); ver bloque ST2.")

    lineas.append("")
    lineas.append("BR4 — mínimos por fila (lectura desde columnas integradas)")
    miss: dict[str, int] = {}
    if COL_ID in df.columns:
        miss["idpel"] = int(es_vacio(df[COL_ID]).sum())
    if COL_TITULO in df.columns:
        miss["titulo"] = int(es_vacio(df[COL_TITULO]).sum())
    if COL_ANO in df.columns:
        y = pd.to_numeric(df[COL_ANO], errors="coerce")
        miss["anoestreno_nulo_o_nan"] = int(y.isna().sum())
    if COL_DIR in df.columns:
        miss["directores"] = int(es_vacio(df[COL_DIR]).sum())
    if COL_ACT in df.columns:
        miss["actores"] = int(es_vacio(df[COL_ACT]).sum())
    if COL_GEN in df.columns:
        miss["generos_vacio"] = int(df[COL_GEN].map(lambda x: genero_esta_vacio(x)).sum())
    for k, v in miss.items():
        lineas.append(f"  {k}: {v} / {n} ({100 * v / max(n, 1):.2f}%)")

    lineas.append("")
    lineas.append(f"BR5 — géneros vs lista válida ({TXT_GENEROS_VALIDOS.name})")
    valid = _leer_generos_validos(TXT_GENEROS_VALIDOS)
    if not valid:
        lineas.append(f"  No se pudo leer {TXT_GENEROS_VALIDOS}")
    elif COL_GEN not in df.columns:
        lineas.append("  Sin columna generos.")
    else:
        mgv = calcular_metricas_generos_lista_valida(df, valid)
        lineas.append(
            f"  Filas con ≥1 género reconocido en la lista: {mgv.get('filas_con_al_menos_un_genero_valido', 0)} "
            f"({float(mgv.get('pct_sobre_total_filas') or 0):.2f}% del total)"
        )
        if mgv.get("pct_sobre_filas_con_genero_no_vacio") is not None:
            lineas.append(
                f"  Sobre filas con género no vacío: "
                f"{float(mgv['pct_sobre_filas_con_genero_no_vacio']):.2f}% "
                f"({mgv.get('filas_con_genero_pero_ningun_valido', 0)} sin ningún término de la taxonomía)"
            )
        total_items = 0
        invalid_items = 0
        for raw in df[COL_GEN].dropna():
            s = str(raw).strip()
            if not s.startswith("["):
                continue
            items = [x.lower() for x in _parse_generos_lista_python(raw)]
            total_items += len(items)
            invalid_items += sum(1 for x in items if x not in valid)
        lineas.append(f"  (Detalle ítems solo formato lista) evaluados: {total_items}; no en lista: {invalid_items}")
        lineas.append(
            "  Criterio filas: lista D2 + texto Netflix con segmentación, sufijos y subcadenas vs genres.txt."
        )

    lineas.append("")
    return "\n".join(lineas)


def construir_hojas_excel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Arma los DataFrames que van a cada pestaña del Excel.

    """
    n = len(df)
    hojas: dict[str, pd.DataFrame] = {}

    _m_res = calcular_metricas_referencias_y_recencia(df)
    pct_gen = _m_res.get("genres_txt_pct_filas_con_alguno_valido")
    hojas["Resumen"] = pd.DataFrame(
        {
            "metrica": [
                "filas",
                "columnas",
                "filas_duplicadas",
                "titulos_repetidos",
                "pct_filas_algún_género_válido_genres_txt",
            ],
            "valor": [
                n,
                len(df.columns),
                int(df.duplicated().sum()),
                int(df[COL_TITULO].duplicated().sum()) if COL_TITULO in df.columns else None,
                round(float(pct_gen), 2) if pct_gen is not None else None,
            ],
        }
    )

    # Faltantes
    filas_falt = []
    for c in df.columns:
        vac = int(es_vacio(df[c]).sum())
        filas_falt.append({"columna": c, "vacios": vac, "pct": round(100.0 * vac / max(n, 1), 2)})
    hojas["Faltantes"] = pd.DataFrame(filas_falt)

    # Año
    if COL_ANO in df.columns:
        y = pd.to_numeric(df[COL_ANO], errors="coerce")
        fuera = (y < ANIO_MIN) | (y > ANIO_MAX)
        hojas["Ano_estreno"] = pd.DataFrame(
            {
                "metrica": [
                    "min",
                    "max",
                    "nulos_o_no_numericos",
                    f"fuera_{ANIO_MIN}_{ANIO_MAX}",
                    f"recientes_>=_{ANIO_REFERENCIA - VENTANA_RECIENTE_ANIOS}",
                ],
                "valor": [
                    y.min(),
                    y.max(),
                    int(y.isna().sum()),
                    int(fuera.sum()),
                    int((y >= (ANIO_REFERENCIA - VENTANA_RECIENTE_ANIOS)).sum()),
                ],
            }
        )
        dec = ((y // 10) * 10).dropna()
        if len(dec):
            hojas["Ano_por_decada"] = (
                dec.astype(int).value_counts().sort_index().rename_axis("decada").reset_index(name="cantidad")
            )

    # Duración (conviven textos tipo "90 min", "2 Seasons" y números sueltos de D2)
    if COL_DUR in df.columns:
        d = df[COL_DUR].dropna().astype(str).str.strip()
        es_min = d.str.contains("min", case=False, na=False)
        es_sea = d.str.contains("season", case=False, na=False)
        es_dig = d.str.match(r"^\d+$", na=False)
        es_otro = ~(es_min | es_sea | es_dig)
        hojas["Duracion_patrones"] = pd.DataFrame(
            {
                "patron": ["con_min", "con_season", "solo_digitos", "resto_sin_clasificar"],
                "cantidad": [
                    int(es_min.sum()),
                    int(es_sea.sum()),
                    int(es_dig.sum()),
                    int(es_otro.sum()),
                ],
            }
        )

    # Géneros formato
    if COL_GEN in df.columns:
        g = df[COL_GEN].dropna().astype(str).str.strip()
        hojas["Generos_formato"] = pd.DataFrame(
            {
                "tipo": ["lista_python_[]", "texto_otro"],
                "cantidad": [int(g.str.startswith("[").sum()), int((~g.str.startswith("[")).sum())],
            }
        )

    # BR / DQ métricas
    br_rows = []
    if COL_TITULO in df.columns:
        br_rows.append(("BR1_sin_titulo", int(es_vacio(df[COL_TITULO]).sum())))
    if COL_ANO in df.columns:
        y = pd.to_numeric(df[COL_ANO], errors="coerce")
        validos = (y >= ANIO_MIN) & (y <= ANIO_MAX)
        br_rows.append(("BR4_anoestreno_invalido_o_nulo", int((~validos).sum())))
    if COL_DIR in df.columns:
        br_rows.append(("BR4_sin_directores", int(es_vacio(df[COL_DIR]).sum())))
    if COL_ACT in df.columns:
        br_rows.append(("BR4_sin_actores", int(es_vacio(df[COL_ACT]).sum())))
    if COL_GEN in df.columns:
        br_rows.append(("BR4_sin_genero_util", int(df[COL_GEN].map(genero_esta_vacio).sum())))
    if COL_DESC in df.columns:
        br_rows.append(("DQ_descripcion_vacia", int(es_vacio(df[COL_DESC]).sum())))
    hojas["Reglas_y_DQ"] = pd.DataFrame(br_rows, columns=["codigo", "filas_afectadas"])

    # Cardinalidad
    card = []
    for col in df.columns:
        try:
            nu = df[col].nunique(dropna=True)
        except TypeError:
            nu = df[col].nunique()
        card.append({"columna": col, "valores_distintos": nu})
    hojas["Cardinalidad"] = pd.DataFrame(card)

    # Top 100, Oscar y % últimos 10 años
    _m = calcular_metricas_referencias_y_recencia(df)
    filas_ref: list[dict[str, Any]] = []
    for clave in sorted(_m.keys()):
        val = _m[clave]
        if val is None or isinstance(val, (str, int, float, bool)):
            filas_ref.append({"metrica": clave, "valor": val})
        else:
            filas_ref.append({"metrica": clave, "valor": str(val)})
    hojas["Referencias_recencia"] = pd.DataFrame(filas_ref)

    _gv = _leer_generos_validos(TXT_GENEROS_VALIDOS)
    _mgv = calcular_metricas_generos_lista_valida(df, _gv)
    filas_gv: list[dict[str, Any]] = []
    for clave in sorted(_mgv.keys()):
        val = _mgv[clave]
        if val is None or isinstance(val, (str, int, float, bool)):
            filas_gv.append({"metrica": clave, "valor": val})
        else:
            filas_gv.append({"metrica": clave, "valor": str(val)})
    hojas["Generos_vs_lista_valida"] = pd.DataFrame(filas_gv)

    return hojas


def escribir_excel(hojas: dict[str, pd.DataFrame], path: Path) -> None:
    """Graba todas las pestañas en un solo xlsx (necesita openpyxl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for nombre, tabla in hojas.items():
            safe = nombre[:31]  # límite Excel
            tabla.to_excel(writer, sheet_name=safe, index=False)


def texto_dimensiones_completo(df: pd.DataFrame) -> str:
    """Junta los bloques de dimensiones + chequeos en un solo .txt."""
    partes = [
        analizar_completitud(df),
        "",
        analizar_unicidad(df),
        "",
        analizar_exactitud_sintactica(df),
        "",
        analizar_frescura(df),
        "",
        profiling_chequeos_extendidos(df),
        verificar_reglas_negocio_extendido(df),
    ]
    return "\n".join(partes)


def main() -> None:
    """
    Corre todo el pipeline: lee BDPel, escribe txts de apoyo y el Excel.

    Si falta openpyxl, igual dejamos los txt y avisamos por consola para instalar.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = resolver_bdpel_csv()
    if not csv_path.is_file():
        print(f"ERROR: no se encontró BDPel.csv en {BASE}", file=sys.stderr)
        raise SystemExit(1)

    df = cargar_bdpel(csv_path)

    (OUT_DIR / "profiling_st2_bdpel.txt").write_text(st2_profiling(df), encoding="utf-8")
    (OUT_DIR / "profiling_br_bdpel.txt").write_text(br_resumen(df), encoding="utf-8")
    (OUT_DIR / "profiling_dimensiones_bdpel.txt").write_text(texto_dimensiones_completo(df), encoding="utf-8")

    xlsx_path = OUT_DIR / "data_profiling_BDPel.xlsx"
    try:
        hojas = construir_hojas_excel(df)
        escribir_excel(hojas, xlsx_path)
    except ImportError:
        print("AVISO: instalá openpyxl para generar el Excel: pip install openpyxl", file=sys.stderr)
    else:
        print(f"Excel: {xlsx_path}")

    print(f"OK: textos y (si hubo openpyxl) xlsx en {OUT_DIR}")


if __name__ == "__main__":
    main()
