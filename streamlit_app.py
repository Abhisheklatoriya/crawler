import os
import re
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
ID_PATTERNS = [
    re.compile(r"\bGI-(\d{6,})\b", re.IGNORECASE),   # GI-1490377075 -> 1490377075
    re.compile(r"\b(\d{6,})\b")                      # fallback: long number anywhere
]


def extract_id_from_text(text: str) -> str | None:
    """Extract a likely asset ID (digits) from text (filename or URL)."""
    if not text:
        return None
    for pat in ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def sanitize_folder_name(name: str) -> str:
    """Make a safe folder name."""
    if not name:
        return "Unknown"
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)  # Windows-unsafe chars
    return name or "Unknown"


def build_id_to_lob_map(df: pd.DataFrame,
                        weblink_col="Web Link",
                        lob_col="LOB") -> tuple[dict, dict]:
    """
    Returns:
      - id_to_lob: {id: lob} (if duplicates, keeps the first non-empty lob)
      - dup_ids: {id: [lob1, lob2, ...]} for diagnostics
    """
    id_to_lob = {}
    dup_ids = defaultdict(list)

    for _, row in df.iterrows():
        link = str(row.get(weblink_col, "")).strip()
        lob = str(row.get(lob_col, "")).strip()

        asset_id = extract_id_from_text(link)
        if not asset_id:
            continue

        if asset_id in id_to_lob:
            dup_ids[asset_id].append(lob)
            # Prefer keeping an existing non-empty lob; otherwise overwrite with non-empty
            if (not id_to_lob[asset_id]) and lob:
                id_to_lob[asset_id] = lob
        else:
            id_to_lob[asset_id] = lob
            dup_ids[asset_id].append(lob)

    return id_to_lob, dup_ids


def human_bytes(n: float) -> str:
    """Human-readable bytes."""
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while n >= 1024 and u < len(units) - 1:
        n /= 1024
        u += 1
    return f"{n:.2f} {units[u]}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LOB Auto Organizer", layout="wide")
st.title("LOB Auto Organizer (Upload one-by-one → auto-sort → ETA)")

with st.expander("How it works", expanded=False):
    st.write(
        """
        1) Upload your sheet (CSV export from Google Sheets).
        2) The app builds a map from Web Link → Asset ID → LOB.
        3) Upload creatives one-by-one. The app reads the ID from the filename and moves/saves it into:
           output/<LOB>/<filename>
        4) ETA is based on your upload speed + the total size you tell the app you plan to upload.
        """
    )

colA, colB = st.columns([1, 1])

with colA:
    total_gb = st.number_input(
        "Total size you expect to upload (GB) — used for ETA",
        min_value=0.0,
        value=45.0,
        step=1.0
    )
with colB:
    output_root = st.text_input("Output folder", value="output")

output_root_path = Path(output_root)
output_root_path.mkdir(parents=True, exist_ok=True)

st.divider()

# --- Step 1: Load CSV
st.subheader("1) Upload the sheet (CSV)")
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")

if "id_to_lob" not in st.session_state:
    st.session_state.id_to_lob = None
    st.session_state.dup_ids = None

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write("Preview:")
    st.dataframe(df.head(15), use_container_width=True)

    # Try common column names; you can change here if your sheet differs
    weblink_col_guess = "Web Link"
    lob_col_guess = "LOB"

    missing = []
    if weblink_col_guess not in df.columns:
        missing.append(weblink_col_guess)
    if lob_col_guess not in df.columns:
        missing.append(lob_col_guess)

    if missing:
        st.error(f"Missing expected columns: {missing}. Rename columns or adjust code.")
        st.stop()

    id_to_lob, dup_ids = build_id_to_lob_map(df, weblink_col_guess, lob_col_guess)
    st.session_state.id_to_lob = id_to_lob
    st.session_state.dup_ids = dup_ids

    st.success(f"Loaded mapping for {len(id_to_lob):,} IDs.")
    dup_count = sum(1 for k, v in dup_ids.items() if len(set([x for x in v if x])) > 1)
    if dup_count:
        st.warning(f"Found {dup_count} IDs mapped to multiple LOB values. App will use first non-empty LOB.")

st.divider()

# --- Tracking state for ETA + logs
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "bytes_done" not in st.session_state:
    st.session_state.bytes_done = 0
if "files_done" not in st.session_state:
    st.session_state.files_done = 0
if "log" not in st.session_state:
    st.session_state.log = []  # list of dict rows

total_bytes = int(total_gb * 1024**3)

# --- Step 2: Upload files one by one
st.subheader("2) Upload files one-by-one (the app will auto-sort into LOB folders)")

if st.session_state.id_to_lob is None:
    st.info("Upload the CSV first so the app knows how to map IDs → LOB.")
    st.stop()

uploaded = st.file_uploader(
    "Upload a creative file (one at a time)",
    type=None,
    accept_multiple_files=False,
    key="file_uploader"
)

# Status area
m1, m2, m3, m4 = st.columns(4)
elapsed = 0.0
if st.session_state.start_time:
    elapsed = time.time() - st.session_state.start_time

speed_bps = (st.session_state.bytes_done / elapsed) if elapsed > 0 else 0.0
remaining_bytes = max(total_bytes - st.session_state.bytes_done, 0)
eta_seconds = (remaining_bytes / speed_bps) if speed_bps > 0 else None

with m1:
    st.metric("Files processed", f"{st.session_state.files_done:,}")
with m2:
    st.metric("Data processed", human_bytes(st.session_state.bytes_done))
with m3:
    st.metric("Avg speed", f"{human_bytes(speed_bps)}/s" if speed_bps else "—")
with m4:
    st.metric("ETA", time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds else "—")

progress = 0.0
if total_bytes > 0:
    progress = min(st.session_state.bytes_done / total_bytes, 1.0)
st.progress(progress)

# --- Process uploaded file
if uploaded is not None:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    filename = uploaded.name
    file_size = uploaded.size  # bytes
    asset_id = extract_id_from_text(filename)

    if not asset_id:
        st.error(f"Could not extract an ID from filename: {filename}")
        st.session_state.log.append({
            "filename": filename,
            "asset_id": None,
            "lob": None,
            "status": "NO_ID_IN_FILENAME",
            "bytes": file_size
        })
    else:
        lob = st.session_state.id_to_lob.get(asset_id, "")
        lob_folder = sanitize_folder_name(lob) if lob else "Needs_Match"

        dest_dir = output_root_path / lob_folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / filename

        # Save file to disk
        with open(dest_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Saved → {dest_path}  (ID: {asset_id}, LOB: {lob_folder})")

        st.session_state.bytes_done += file_size
        st.session_state.files_done += 1
        st.session_state.log.append({
            "filename": filename,
            "asset_id": asset_id,
            "lob": lob_folder,
            "status": "SAVED",
            "bytes": file_size
        })

    st.rerun()

st.divider()

st.subheader("Run log")
log_df = pd.DataFrame(st.session_state.log)
st.dataframe(log_df.tail(50), use_container_width=True)

if not log_df.empty:
    st.download_button(
        "Download log as CSV",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="lob_organizer_log.csv",
        mime="text/csv"
    )

st.caption("Tip: Files without a match go into output/Needs_Match so you can review them later.")

