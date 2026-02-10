# streamlit_app.py
# ------------------------------------------------------------
# LOB Auto Organizer
# - Upload CSV (sheet export)
# - Upload image files one-by-one
# - Extract ID FROM IMAGE FILENAME
# - Match that ID to ID extracted from the sheet link column
# - Save into output/<LOB>/<filename> (only allowed LOBs)
# - Provide ETA based on expected total GB
# - Build ZIP(s) for download at the end
# ------------------------------------------------------------

import re
import time
import zipfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st


# -----------------------------
# Allowed LOBs (from your dropdown)
# -----------------------------
ALLOWED_LOBS = ["Wireless", "Home", "Business", "Brand", "Bank"]
ALLOWED_LOBS_LOWER = {x.lower(): x for x in ALLOWED_LOBS}  # canonical mapping
UNMATCHED_DIR = "Needs_Match"


def normalize_lob(lob_raw: str) -> str | None:
    """Return canonical allowed LOB name if valid, else None."""
    if lob_raw is None:
        return None
    lob = str(lob_raw).strip()
    if not lob:
        return None
    return ALLOWED_LOBS_LOWER.get(lob.lower())


# -----------------------------
# ID extraction
# -----------------------------
# IMPORTANT: Matching uses ID extracted FROM IMAGE FILENAME
GI_IN_NAME = re.compile(r"GI-(\d{6,})", re.IGNORECASE)
GETTY_LICENSE_IN_NAME = re.compile(r"license[_-]?(\d{6,})", re.IGNORECASE)
ANY_LONG_NUM = re.compile(r"(\d{6,})")

# For sheet links (web link), we typically just need a long number, or GI-#### if present
GI_IN_TEXT = re.compile(r"\bGI-(\d{6,})\b", re.IGNORECASE)
ANY_LONG_NUM_IN_TEXT = re.compile(r"\b(\d{6,})\b")


def extract_id_from_filename(filename: str) -> str | None:
    """
    Extract ID strictly from filename (your requirement).
    Priority:
      1) GI-########
      2) license_######## or license-########
      3) fallback: choose the longest long-number sequence in the filename
    """
    if not filename:
        return None

    m = GI_IN_NAME.search(filename)
    if m:
        return m.group(1)

    m = GETTY_LICENSE_IN_NAME.search(filename)
    if m:
        return m.group(1)

    nums = ANY_LONG_NUM.findall(filename)
    if not nums:
        return None

    best_len = max(len(n) for n in nums)
    candidates = [n for n in nums if len(n) == best_len]
    return candidates[-1] if candidates else None


def extract_id_from_link(text: str) -> str | None:
    """
    Extract ID from sheet link (Web Link / Box Link).
    Priority:
      1) GI-########
      2) fallback: choose the longest long-number sequence in the link
    """
    if not text:
        return None

    m = GI_IN_TEXT.search(text)
    if m:
        return m.group(1)

    nums = ANY_LONG_NUM_IN_TEXT.findall(text)
    if not nums:
        return None

    best_len = max(len(n) for n in nums)
    candidates = [n for n in nums if len(n) == best_len]
    return candidates[-1] if candidates else None


def human_bytes(n: float) -> str:
    """Human-readable bytes."""
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while n >= 1024 and u < len(units) - 1:
        n /= 1024
        u += 1
    return f"{n:.2f} {units[u]}"


# -----------------------------
# CSV header detection
# -----------------------------
def read_csv_with_detected_header(uploaded_file) -> pd.DataFrame:
    """
    Your CSV sometimes has the real header row not at row 0.
    Detect the header row by finding a cell with 'Web Link' (case-insensitive).
    """
    raw = pd.read_csv(uploaded_file, header=None, dtype=str, keep_default_na=False)

    header_row_idx = None
    for i in range(min(50, len(raw))):
        row_vals = [str(x).strip() for x in raw.iloc[i].tolist()]
        if any(v.lower() == "web link" for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        header_row_idx = 0

    header = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = [c if c else f"col_{idx}" for idx, c in enumerate(header)]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(how="all")
    return df


# -----------------------------
# Build mapping from sheet: (asset_id -> LOB)
# -----------------------------
def build_id_to_lob_map(df: pd.DataFrame, link_col: str, lob_col: str) -> tuple[dict, dict, int]:
    """
    Returns:
      - id_to_lob: {id: canonical_lob or ""} (canonicalized to ALLOWED_LOBS; empty if invalid/missing)
      - dup_ids: {id: [lob_values...]} diagnostics
      - missing_id_count: rows where we couldn't extract an ID from link_col
    """
    id_to_lob = {}
    dup_ids = defaultdict(list)
    missing_id_count = 0

    for _, row in df.iterrows():
        link = str(row.get(link_col, "")).strip()
        lob_raw = row.get(lob_col, "")

        asset_id = extract_id_from_link(link)
        if not asset_id:
            missing_id_count += 1
            continue

        canonical_lob = normalize_lob(lob_raw) or ""  # empty if not allowed

        if asset_id in id_to_lob:
            dup_ids[asset_id].append(canonical_lob)
            # keep first non-empty canonical lob
            if (not id_to_lob[asset_id]) and canonical_lob:
                id_to_lob[asset_id] = canonical_lob
        else:
            id_to_lob[asset_id] = canonical_lob
            dup_ids[asset_id].append(canonical_lob)

    return id_to_lob, dup_ids, missing_id_count


# -----------------------------
# Zipping helpers
# -----------------------------
def zip_folder(folder_path: Path, zip_path: Path) -> None:
    """Zip the contents of folder_path into zip_path."""
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder_path)))


def zip_matched_only(output_root: Path, zip_path: Path) -> None:
    """Create a zip that contains only matched LOB folders (excludes Needs_Match)."""
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for lob in ALLOWED_LOBS:
            lob_dir = output_root / lob
            if not lob_dir.exists():
                continue
            for p in lob_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(output_root)))


def zip_per_lob(output_root: Path, zips_dir: Path) -> list[Path]:
    """Create one zip per allowed LOB folder. Returns list of zip file paths."""
    zips_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for lob in ALLOWED_LOBS:
        lob_dir = output_root / lob
        if not lob_dir.exists():
            continue
        # Only zip if it has files
        has_file = any(p.is_file() for p in lob_dir.rglob("*"))
        if not has_file:
            continue

        zip_path = zips_dir / f"{lob}.zip"
        zip_folder(lob_dir, zip_path)
        created.append(zip_path)

    return created


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="LOB Auto Organizer", layout="wide")
st.title("LOB Auto Organizer (Filename-based matching + ZIP downloads)")

st.caption(f"Allowed LOBs: {', '.join(ALLOWED_LOBS)}")

col1, col2 = st.columns([1, 1])
with col1:
    total_gb = st.number_input(
        "Total size you expect to upload (GB) — used for ETA",
        min_value=0.0,
        value=45.0,
        step=1.0,
    )
with col2:
    output_root = st.text_input("Output folder", value="output")

output_root_path = Path(output_root)
output_root_path.mkdir(parents=True, exist_ok=True)

# Ensure destination folders exist (exactly like your dropdown)
for lob in ALLOWED_LOBS:
    (output_root_path / lob).mkdir(parents=True, exist_ok=True)
(output_root_path / UNMATCHED_DIR).mkdir(parents=True, exist_ok=True)

st.divider()

# Session state defaults
if "id_to_lob" not in st.session_state:
    st.session_state.id_to_lob = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "bytes_done" not in st.session_state:
    st.session_state.bytes_done = 0
if "files_done" not in st.session_state:
    st.session_state.files_done = 0
if "log" not in st.session_state:
    st.session_state.log = []
if "zip_paths" not in st.session_state:
    st.session_state.zip_paths = []

total_bytes = int(total_gb * 1024**3)

# -----------------------------
# 1) Upload CSV
# -----------------------------
st.subheader("1) Upload the sheet (CSV)")
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")

if csv_file is not None:
    df = read_csv_with_detected_header(csv_file)

    st.write("Preview (detected headers):")
    st.dataframe(df.head(15), use_container_width=True)

    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]

    def find_col(name: str) -> int:
        return lower_cols.index(name.lower()) if name.lower() in lower_cols else 0

    link_col = st.selectbox(
        "Column that contains the link with the ID (usually Web Link)",
        options=cols,
        index=find_col("Web Link"),
    )
    lob_col = st.selectbox(
        "Column that contains LOB",
        options=cols,
        index=find_col("LOB"),
    )

    id_to_lob, dup_ids, missing_id_count = build_id_to_lob_map(df, link_col, lob_col)
    st.session_state.id_to_lob = id_to_lob

    st.success(f"Loaded mapping for {len(id_to_lob):,} IDs.")
    if missing_id_count:
        st.warning(f"{missing_id_count:,} row(s) had no detectable ID in the selected link column.")

    invalid_lob_count = sum(1 for v in id_to_lob.values() if v == "")
    if invalid_lob_count:
        st.warning(
            f"{invalid_lob_count:,} mapped IDs have missing/invalid LOB (not one of {ALLOWED_LOBS}). "
            f"Those uploads will go to {UNMATCHED_DIR}."
        )

    dup_count = sum(1 for _, v in dup_ids.items() if len(set([x for x in v if x])) > 1)
    if dup_count:
        st.warning(
            f"Found {dup_count} ID(s) mapped to multiple LOB values. "
            f"The app keeps the first non-empty LOB."
        )

st.divider()

# -----------------------------
# 2) Upload files one-by-one
# -----------------------------
st.subheader("2) Upload image files one-by-one (matched by FILENAME → LOB)")

if st.session_state.id_to_lob is None:
    st.info("Upload the CSV first so the app knows how to map IDs → LOB.")
    st.stop()

elapsed = 0.0
if st.session_state.start_time:
    elapsed = time.time() - st.session_state.start_time

speed_bps = (st.session_state.bytes_done / elapsed) if elapsed > 0 else 0.0
remaining_bytes = max(total_bytes - st.session_state.bytes_done, 0)
eta_seconds = (remaining_bytes / speed_bps) if speed_bps > 0 else None

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Files processed", f"{st.session_state.files_done:,}")
with m2:
    st.metric("Data processed", human_bytes(st.session_state.bytes_done))
with m3:
    st.metric("Avg speed", f"{human_bytes(speed_bps)}/s" if speed_bps else "—")
with m4:
    st.metric("ETA", time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds else "—")

progress = min(st.session_state.bytes_done / total_bytes, 1.0) if total_bytes > 0 else 0.0
st.progress(progress)

uploaded = st.file_uploader(
    "Upload an image (one at a time)",
    accept_multiple_files=False,
    key="file_uploader",
)

if uploaded is not None:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    filename = uploaded.name
    file_size = uploaded.size

    # ✅ Extract ID from the IMAGE FILE NAME (your requirement)
    asset_id = extract_id_from_filename(filename)

    if not asset_id:
        st.error(f"Could not extract an ID from the filename: {filename}")
        dest_folder = UNMATCHED_DIR
        st.session_state.log.append(
            {"filename": filename, "asset_id": None, "lob": dest_folder, "status": "NO_ID_IN_FILENAME", "bytes": file_size}
        )
    else:
        mapped_lob = st.session_state.id_to_lob.get(asset_id, "")
        dest_folder = mapped_lob if mapped_lob in ALLOWED_LOBS else UNMATCHED_DIR

        dest_dir = output_root_path / dest_folder
        dest_path = dest_dir / filename

        with open(dest_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Saved → {dest_path}   (Filename ID: {asset_id}, LOB: {dest_folder})")

        st.session_state.bytes_done += file_size
        st.session_state.files_done += 1
        st.session_state.log.append(
            {"filename": filename, "asset_id": asset_id, "lob": dest_folder, "status": "SAVED", "bytes": file_size}
        )

    # output changed => clear any previous zips
    st.session_state.zip_paths = []
    st.rerun()

st.divider()

# -----------------------------
# 3) Build ZIPs + Download
# -----------------------------
st.subheader("3) Download (build ZIP when you're done uploading)")
st.write("Once uploads are complete, build ZIP(s) so the matched creatives are ready to download.")

zips_dir = output_root_path / "_zips"
zips_dir.mkdir(parents=True, exist_ok=True)

c1, c2, c3 = st.columns([1, 1, 1])
build_one_zip = c1.button("Build ONE ZIP (matched only)")
build_lob_zips = c2.button("Build ZIP per LOB (matched only)")
build_unmatched_zip = c3.button(f"Build ZIP for {UNMATCHED_DIR}")

if build_one_zip:
    with st.spinner("Building matched_creatives.zip ..."):
        zip_path = zips_dir / "matched_creatives.zip"
        zip_matched_only(output_root_path, zip_path)
        st.session_state.zip_paths = [zip_path]
    st.success("ZIP created: matched_creatives.zip")

if build_lob_zips:
    with st.spinner("Building ZIPs per LOB ..."):
        created = zip_per_lob(output_root_path, zips_dir)
        st.session_state.zip_paths = created
    st.success(f"Created {len(st.session_state.zip_paths)} ZIP(s).")

if build_unmatched_zip:
    with st.spinner(f"Building {UNMATCHED_DIR}.zip ..."):
        zip_path = zips_dir / f"{UNMATCHED_DIR}.zip"
        zip_folder(output_root_path / UNMATCHED_DIR, zip_path)
        existing = [p for p in st.session_state.zip_paths if isinstance(p, Path) and p.exists()]
        st.session_state.zip_paths = existing + [zip_path]
    st.success(f"ZIP created: {UNMATCHED_DIR}.zip")

existing_zips = [p for p in st.session_state.zip_paths if isinstance(p, Path) and p.exists()]

if existing_zips:
    st.markdown("### Download ZIP(s)")
    for zp in existing_zips:
        st.download_button(
            label=f"Download {zp.name} ({human_bytes(zp.stat().st_size)})",
            data=zp.read_bytes(),
            file_name=zp.name,
            mime="application/zip",
            key=f"dl_{zp.name}",
        )
else:
    st.info("No ZIP built yet. Upload files first, then click a Build ZIP button.")

st.divider()

# -----------------------------
# Log
# -----------------------------
st.subheader("Run log")
log_df = pd.DataFrame(st.session_state.log)
st.dataframe(log_df.tail(200), use_container_width=True)

if not log_df.empty:
    st.download_button(
        "Download log as CSV",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="lob_organizer_log.csv",
        mime="text/csv",
    )

st.caption(f"Unmatched files are saved into {output_root}/{UNMATCHED_DIR}/")
