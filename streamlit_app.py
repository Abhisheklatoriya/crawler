# streamlit_app.py
import re
import time
import zipfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st


# -----------------------------
# ID extraction (filename + URL)
# -----------------------------
GI_ID_RE = re.compile(r"\bGI-(\d{6,})\b", re.IGNORECASE)
LONG_NUM_RE = re.compile(r"\b(\d{6,})\b")  # fallback: any long number


def extract_id_from_text(text: str) -> str | None:
    """Extract an asset id (digits) from filename or URL."""
    if not text:
        return None

    m = GI_ID_RE.search(text)
    if m:
        return m.group(1)

    nums = LONG_NUM_RE.findall(text)
    if not nums:
        return None

    # Heuristic: choose the longest number; if tie, pick the last occurrence
    best_len = max(len(n) for n in nums)
    candidates = [n for n in nums if len(n) == best_len]
    return candidates[-1] if candidates else nums[-1]


def sanitize_folder_name(name: str) -> str:
    """Make a safe folder name."""
    if not name:
        return "Unknown"
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name or "Unknown"


def human_bytes(n: float) -> str:
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
    Reads CSV where the "real" header row might not be row 0.
    Detects the row containing "Web Link" (case-insensitive) and uses it as header.
    """
    raw = pd.read_csv(uploaded_file, header=None, dtype=str, keep_default_na=False)

    header_row_idx = None
    search_rows = min(50, len(raw))
    for i in range(search_rows):
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
def build_id_to_lob_map(df: pd.DataFrame, link_col: str, lob_col: str) -> tuple[dict, dict]:
    """
    Returns:
      - id_to_lob: {id: lob} (if duplicates, keeps the first non-empty lob)
      - dup_ids: {id: [lob1, lob2, ...]} for diagnostics
    """
    id_to_lob = {}
    dup_ids = defaultdict(list)

    for _, row in df.iterrows():
        link = str(row.get(link_col, "")).strip()
        lob = str(row.get(lob_col, "")).strip()

        asset_id = extract_id_from_text(link)
        if not asset_id:
            continue

        if asset_id in id_to_lob:
            dup_ids[asset_id].append(lob)
            if (not id_to_lob[asset_id]) and lob:
                id_to_lob[asset_id] = lob
        else:
            id_to_lob[asset_id] = lob
            dup_ids[asset_id].append(lob)

    return id_to_lob, dup_ids


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
                # store relative paths inside zip
                zf.write(p, arcname=str(p.relative_to(folder_path)))


def zip_matched_only(output_root: Path, zip_path: Path, unmatched_folder_name="Needs_Match") -> None:
    """
    Create a zip that contains ONLY matched LOB folders (excludes Needs_Match).
    """
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for lob_dir in output_root.iterdir():
            if not lob_dir.is_dir():
                continue
            if lob_dir.name == unmatched_folder_name:
                continue
            for p in lob_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(output_root)))


def zip_per_lob(output_root: Path, zips_dir: Path, unmatched_folder_name="Needs_Match") -> list[Path]:
    """
    Create one zip per LOB folder (excludes Needs_Match). Returns list of zip file paths.
    """
    zips_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for lob_dir in output_root.iterdir():
        if not lob_dir.is_dir():
            continue
        if lob_dir.name == unmatched_folder_name:
            continue

        zip_path = zips_dir / f"{lob_dir.name}.zip"
        zip_folder(lob_dir, zip_path)
        created.append(zip_path)

    return created


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="LOB Auto Organizer", layout="wide")
st.title("LOB Auto Organizer (Upload one-by-one → auto-sort → ZIP download at end)")

with st.expander("What this app does", expanded=False):
    st.write(
        """
        - Upload your sheet as CSV (export from Google Sheets).
        - App builds a mapping: Link (contains ID) → ID → LOB.
        - Upload creative files one-by-one:
          - Extract ID from filename
          - Find LOB from mapping
          - Save into output/<LOB>/<filename> (matched) OR output/Needs_Match (unmatched)
        - When finished, generate ZIPs ready to download.
        """
    )

col1, col2 = st.columns([1, 1])
with col1:
    total_gb = st.number_input("Total size you expect to upload (GB) — used for ETA", min_value=0.0, value=45.0, step=1.0)
with col2:
    output_root = st.text_input("Output folder", value="output")

output_root_path = Path(output_root)
output_root_path.mkdir(parents=True, exist_ok=True)

UNMATCHED_DIR = "Needs_Match"
(output_root_path / UNMATCHED_DIR).mkdir(parents=True, exist_ok=True)

st.divider()

# Session state defaults
if "id_to_lob" not in st.session_state:
    st.session_state.id_to_lob = None
if "dup_ids" not in st.session_state:
    st.session_state.dup_ids = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "bytes_done" not in st.session_state:
    st.session_state.bytes_done = 0
if "files_done" not in st.session_state:
    st.session_state.files_done = 0
if "log" not in st.session_state:
    st.session_state.log = []
if "zip_paths" not in st.session_state:
    st.session_state.zip_paths = []  # store generated zip file paths


total_bytes = int(total_gb * 1024**3)

# -----------------------------
# Step 1: Upload CSV
# -----------------------------
st.subheader("1) Upload the sheet (CSV)")
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")

if csv_file is not None:
    df = read_csv_with_detected_header(csv_file)

    st.write("Preview (detected headers):")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("### Select the correct columns")
    cols = list(df.columns)

    def find_col(name: str) -> int:
        lower = [c.lower() for c in cols]
        return lower.index(name.lower()) if name.lower() in lower else 0

    link_col = st.selectbox("Column that contains the link with the ID (usually Web Link)", options=cols, index=find_col("Web Link"))
    lob_col = st.selectbox("Column that contains LOB", options=cols, index=find_col("LOB"))

    id_to_lob, dup_ids = build_id_to_lob_map(df, link_col, lob_col)
    st.session_state.id_to_lob = id_to_lob
    st.session_state.dup_ids = dup_ids

    st.success(f"Loaded mapping for {len(id_to_lob):,} IDs.")

    dup_count = sum(1 for _, v in dup_ids.items() if len(set([x for x in v if x])) > 1)
    if dup_count:
        st.warning(
            f"Found {dup_count} IDs mapped to multiple LOB values. "
            f"The app will keep the first non-empty LOB for each ID."
        )

st.divider()

# -----------------------------
# Step 2: Upload files one-by-one
# -----------------------------
st.subheader("2) Upload files one-by-one (auto-sorts into LOB folders)")

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

uploaded = st.file_uploader("Upload a creative file (one at a time)", accept_multiple_files=False, key="file_uploader")

if uploaded is not None:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    filename = uploaded.name
    file_size = uploaded.size
    asset_id = extract_id_from_text(filename)

    if not asset_id:
        st.error(f"Could not extract an ID from filename: {filename}")
        st.session_state.log.append({"filename": filename, "asset_id": None, "lob": None, "status": "NO_ID_IN_FILENAME", "bytes": file_size})
    else:
        lob_raw = st.session_state.id_to_lob.get(asset_id, "")
        lob_folder = sanitize_folder_name(lob_raw) if lob_raw else UNMATCHED_DIR

        dest_dir = output_root_path / lob_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        with open(dest_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Saved → {dest_path}   (ID: {asset_id}, LOB: {lob_folder})")

        st.session_state.bytes_done += file_size
        st.session_state.files_done += 1
        st.session_state.log.append({"filename": filename, "asset_id": asset_id, "lob": lob_folder, "status": "SAVED", "bytes": file_size})

    # clear ZIPs because output changed
    st.session_state.zip_paths = []
    st.rerun()

st.divider()

# -----------------------------
# Step 3: Build ZIPs + Download
# -----------------------------
st.subheader("3) Download (ready once uploads are done)")

st.write("When you finish uploading, create a ZIP to download the organized folders.")

btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

with btn_col1:
    build_one_zip = st.button("Build ONE ZIP (matched only)")
with btn_col2:
    build_lob_zips = st.button("Build ZIP per LOB (matched only)")
with btn_col3:
    build_unmatched_zip = st.button("Build ZIP for Needs_Match")

zips_dir = output_root_path / "_zips"
zips_dir.mkdir(parents=True, exist_ok=True)

if build_one_zip:
    with st.spinner("Building matched_creatives.zip ..."):
        zip_path = zips_dir / "matched_creatives.zip"
        zip_matched_only(output_root_path, zip_path, unmatched_folder_name=UNMATCHED_DIR)
        st.session_state.zip_paths = [zip_path]
    st.success("ZIP created: matched_creatives.zip")

if build_lob_zips:
    with st.spinner("Building ZIPs per LOB ..."):
        created = zip_per_lob(output_root_path, zips_dir, unmatched_folder_name=UNMATCHED_DIR)
        st.session_state.zip_paths = created
    st.success(f"Created {len(st.session_state.zip_paths)} ZIP(s).")

if build_unmatched_zip:
    with st.spinner("Building needs_match.zip ..."):
        zip_path = zips_dir / "needs_match.zip"
        zip_folder(output_root_path / UNMATCHED_DIR, zip_path)
        # keep any already-created zips + add this
        existing = [p for p in st.session_state.zip_paths if p.exists()]
        st.session_state.zip_paths = existing + [zip_path]
    st.success("ZIP created: needs_match.zip")

# Download buttons
existing_zips = [p for p in st.session_state.zip_paths if isinstance(p, Path) and p.exists()]

if existing_zips:
    st.markdown("### Download ZIP(s)")
    for zp in existing_zips:
        # NOTE: Streamlit download_button requires bytes; for large zips, this will use memory.
        # This is why "ZIP per LOB" is often safer for big totals.
        zip_bytes = zp.read_bytes()
        st.download_button(
            label=f"Download {zp.name} ({human_bytes(zp.stat().st_size)})",
            data=zip_bytes,
            file_name=zp.name,
            mime="application/zip",
            key=f"dl_{zp.name}",
        )
else:
    st.info("No ZIP built yet. Upload files first, then click one of the Build ZIP buttons.")

st.divider()

# -----------------------------
# Log + export
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

st.caption(f"Unmatched files are saved into output/{UNMATCHED_DIR}/ so you can review them later.")
