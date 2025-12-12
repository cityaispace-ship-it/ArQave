import os, json, shutil, subprocess, re
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import camelot

def has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run_ocrmypdf(in_pdf: Path, out_pdf: Path) -> None:
    # OCR normalization; never adds "n.d", just fixes page quality
    cmd = [
        "ocrmypdf",
        "--deskew",
        "--rotate-pages",
        "--remove-background",
        "--optimize", "3",
        "--force-ocr",
        str(in_pdf), str(out_pdf)
    ]
    subprocess.run(cmd, check=True)


ND_TOKENS = {"n.d", "n.d.", "nd", "n/a", "na", "n.a.", "-", "—", "–", "…", ".", "..", "nil", "none"}

def _normalize_cell(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    # unify unicode dashes and spaces
    s = s.replace("\u2014", "-").replace("\u2013", "-").replace("\u00a0", " ")
    # never introduce "n.d"; only remove known placeholder-ish tokens to blank
    sl = s.lower()
    if sl in ND_TOKENS:
        return ""
    # trim trailing dot after numbers: "12." -> "12"
    if s.endswith("."):
        core = s[:-1]
        if core.replace(",", "").replace(".", "").isdigit():
            return core
    return s

def normalize_df_tokens(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.map(_normalize_cell)  # Updated from applymap (deprecated in pandas 2.1+)
    return out

def try_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = pd.to_numeric(out[c], errors="coerce")
        base_non_null = out[c].notna().sum()
        num_non_null = s.notna().sum()
        if base_non_null > 0 and num_non_null / base_non_null >= 0.5:
            out[c] = s
    return out


NOISE_RE = re.compile(r"^[\s\-\—\–\.\,;/\\|:]+$")  # punctuation/whitespace only

def passes_basic_table_filters(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "empty"
    rows, cols = df.shape
    if rows < 2 or cols < 2:
        return False, "too_small"

    # sparsity: treat "" as NA
    empty_ratio = df.replace("", None).isna().mean().mean()
    if empty_ratio > 0.40:
        return False, "too_sparse"

    # punctuation-only noise: if most non-empty cells are just punctuation/dashes
    non_empty = df.replace("", None).stack()
    if len(non_empty) > 0:
        noise = sum(bool(NOISE_RE.match(str(x))) for x in non_empty)
        if noise / len(non_empty) > 0.30:
            return False, "punctuation_noise"

    return True, "ok"


def tabula_extract_csvs(pdf_path: Path, out_dir: Path, pages: str = "all") -> List[Path]:
    created: List[Path] = []
    if not has_cmd("java"):
        return created
    out_csv = out_dir / "tabula_all.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tabula
        
        tabula.convert_into(str(pdf_path), str(out_csv), output_format="csv", pages=pages, guess=False)
        if out_csv.exists() and out_csv.stat().st_size > 0:
            created.append(out_csv)
    except Exception:
        try:
            subprocess.run(
                ["java", "-jar", "tabula.jar", "-p", pages, "-f", "CSV", "-o", str(out_csv), str(pdf_path)],
                check=True
            )
            if out_csv.exists() and out_csv.stat().st_size > 0:
                created.append(out_csv)
        except Exception:
            pass
    return created

def camelot_lattice_tables(pdf_path: Path, pages: str = "all", line_scale: Optional[int] = 50) -> List[pd.DataFrame]:
    kwargs = dict(pages=pages, flavor="lattice")
    if line_scale is not None:
        kwargs["line_scale"] = line_scale
    try:
        tables = camelot.read_pdf(str(pdf_path), **kwargs)
        return [t.df for t in tables] if tables and tables.n > 0 else []
    except Exception:
        return []


def extract_pdf_with_ocr(
    job_dir: Path,
    pdf_path: Path,
    *,
    mode: str = "excel",   # "excel" or "zip"
    pages: str = "all",
) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = job_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    manifest = {
        "mode": mode,
        "steps": [],
        "tables_kept": 0,
        "tables_dropped": 0,
        "drop_reasons": {},
        "paths": {}
    }

    # 1) OCR normalization
    ocred = job_dir / "ocred.pdf"
    if has_cmd("ocrmypdf"):
        run_ocrmypdf(pdf_path, ocred)
        manifest["steps"].append("ocr")
        input_pdf = ocred
    else:
        manifest["steps"].append("no_ocr_binary_on_path")
        input_pdf = pdf_path

    # 2) Tabula first (conservative)
    kept: List[pd.DataFrame] = []
    dropped = 0
    drop_reasons = {}

    tabula_csvs = tabula_extract_csvs(input_pdf, tables_dir, pages=pages)
    if tabula_csvs:
        for p in tabula_csvs:
            try:
                df = pd.read_csv(p)
                df = normalize_df_tokens(df)          # clean, but do not add "n.d"
                ok, reason = passes_basic_table_filters(df)
                if not ok:
                    dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                else:
                    df = try_cast_numeric(df)          # restore numeric types
                    kept.append(df)
            except Exception:
                dropped += 1
                drop_reasons["read_error"] = drop_reasons.get("read_error", 0) + 1

    # 3) Camelot lattice fallback
    if not kept:
        manifest["steps"].append("camelot_lattice_fallback")
        cdfs = camelot_lattice_tables(input_pdf, pages=pages, line_scale=50)
        for df in cdfs:
            df = normalize_df_tokens(df)
            ok, reason = passes_basic_table_filters(df)
            if not ok:
                dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                df = try_cast_numeric(df)
                kept.append(df)

    # 4) Save results (no fillna with "n.d"; keep blanks)
    csv_paths: List[Path] = []
    for i, df in enumerate(kept, start=1):
        cp = tables_dir / f"table_{i:04d}.csv"
        df.to_csv(cp, index=False)
        csv_paths.append(cp)

    manifest["tables_kept"] = len(kept)
    manifest["tables_dropped"] = dropped
    manifest["drop_reasons"] = drop_reasons

    if mode == "excel":
        out_xlsx = job_dir / "extracted.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            if kept:
                # Write extracted tables
                for i, df in enumerate(kept, start=1):
                    df.to_excel(writer, index=False, sheet_name=f"Table_{i}")
                # Write combined sheet
                pd.concat(kept, ignore_index=True).to_excel(writer, index=False, sheet_name="Combined")
            else:
                # No tables extracted - create placeholder sheet
                placeholder_df = pd.DataFrame({
                    "Message": ["No tables were extracted from this PDF"],
                    "Reason": ["All detected tables failed quality filters"],
                    "Suggestion": ["Try a different PDF or check the manifest.json for details"]
                })
                placeholder_df.to_excel(writer, index=False, sheet_name="No_Tables_Found")
        result_path = out_xlsx
    else:
        import zipfile
        zip_path = job_dir / "tables.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in csv_paths:
                z.write(p, arcname=p.name)
        result_path = zip_path

    manifest["paths"]["result"] = str(result_path)
    manifest["paths"]["tables_dir"] = str(tables_dir)
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    return {
        "tables": len(kept), 
        "result_path": str(result_path), 
        "mode": mode,
        "tables_dropped": dropped,
        "drop_reasons": drop_reasons
    }
