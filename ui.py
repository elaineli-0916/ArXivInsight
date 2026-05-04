import streamlit as st
import streamlit.components.v1 as components
import os
from pathlib import Path
from datetime import datetime
import time
import importlib.util
import sys
import json
import requests
import re  # Added for parsing title if needed
from typing import List

# Import backend modules
from back_end.similarity_calculate import similarity_calculate
from back_end.classify import classify
from back_end.dataprocess import data_process

BASE_DIR = Path(__file__).parent
CRAWL_FILE = BASE_DIR / "crawl.py"
DATES_FILE = BASE_DIR / "crawl_data" / "dates.txt"
IMG_DIR = "./crawl_data/similarity_results"

# Directories to keep but empty
SUB_DIRS = ["classified", "refined", "refined_output", "similarity_results"]


class ImageLoader:
    def __init__(self, folder: str):
        self.folder = Path(folder)

    def list_pngs(self) -> List[Path]:
        if not self.folder.exists():
            return []

        pngs = list(self.folder.rglob("*.png"))

        def _relkey(p: Path) -> str:
            return str(p.relative_to(self.folder)).lower()

        all_first = [p for p in pngs if p.name.lower().startswith("all_")]
        normal = [p for p in pngs if not p.name.lower().startswith("all_")]

        all_first = sorted(all_first, key=_relkey)
        normal = sorted(normal, key=_relkey)

        return all_first + normal


st.set_page_config(page_title="Web Data Crawling and Analysis: arXiv CS.CV", layout="wide")

# ====================================================================
# CUSTOM CSS
# ====================================================================
st.markdown("""
    <style>
        /* 1. Sidebar: Wider default */
        [data-testid="stSidebar"] {
            min-width: 350px !important; 
            width: 25vw; 
        }

        /* 2. Fonts: 1.5x scaling */
        html, body, [class*="css"]  {
            font-size: 18px !important; 
        }

        /* Headers */
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 2.0rem !important; }
        h3 { font-size: 1.75rem !important; }

        /* Buttons */
        .stButton button {
            font-size: 1.2rem !important;
            height: 2.8em !important;
        }

        /* Inputs */
        .stTextInput input, .stSelectbox div, .stDateInput input {
            font-size: 1.1rem !important;
            min-height: 2.8em !important;
        }

        /* Markdown Text/Labels */
        p, label {
            font-size: 1.1rem !important; 
        }

        /* Clean Expanders */
        .streamlit-expanderHeader {
            font-size: 1.2rem !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Helper: Cleanup Function
# -----------------------------
def clean_previous_run():
    root_dir = BASE_DIR / "crawl_data"
    if not root_dir.exists():
        return

    for item in root_dir.iterdir():
        if item.is_file() and item.name != "dates.txt":
            try:
                os.remove(item)
            except Exception as e:
                print(f"Error deleting {item}: {e}")

    for sub in SUB_DIRS:
        sub_path = root_dir / sub
        if sub_path.exists():
            for sub_item in sub_path.iterdir():
                if sub_item.is_file():
                    try:
                        os.remove(sub_item)
                    except Exception as e:
                        print(f"Error deleting {sub_item}: {e}")


# -----------------------------
# Helper: Load Raw Data (FIXED for Dictionary Structure)
# -----------------------------
def load_all_raw_arxiv_data(date_list):
    """
    Iterates through ALL selected dates and loads their individual JSON files.
    Handles the structure: { "metadata": ..., "papers": [...] }
    """
    combined_data = []
    if not date_list:
        return combined_data

    for d in date_list:
        raw_path = BASE_DIR / "crawl_data" / f"arxiv_{d}.json"
        if raw_path.exists():
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Logic: If it's a dict with 'papers', get that list.
                    if isinstance(data, dict) and "papers" in data:
                        combined_data.extend(data["papers"])
                    elif isinstance(data, list):
                        combined_data.extend(data)
            except Exception as e:
                print(f"Error loading raw data for {d}: {e}")

    return combined_data


# -----------------------------
# Module Imports
# -----------------------------
def import_crawl_module():
    crawl_file = os.path.join(os.path.dirname(__file__), "back_end/crawl.py")
    spec = importlib.util.spec_from_file_location("crawl", crawl_file)
    crawl_module = importlib.util.module_from_spec(spec)
    sys.modules["crawl"] = crawl_module
    spec.loader.exec_module(crawl_module)
    return crawl_module


# -----------------------------
# Date Management
# -----------------------------
def read_dates_from_txt():
    if not DATES_FILE.exists():
        return []
    try:
        lines = DATES_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    res = []
    for s in lines:
        s = s.strip()
        if not s: continue
        try:
            datetime.strptime(s, "%Y-%m-%d")
            res.append(s)
        except Exception:
            continue
    return sorted(list(set(res)))


def write_dates_to_txt(dates):
    dates = sorted(list(set(dates)))
    content = ""
    if dates:
        content = "\n".join(dates) + "\n"
    tmp_file = DATES_FILE.with_suffix(".tmp")
    try:
        DATES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tmp_file.open("w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp_file.replace(DATES_FILE)
    except Exception as e:
        st.error(f"Error saving dates: {e}")


def build_dates_label_from_session():
    dates = st.session_state.get("selected_dates", [])
    if not dates: return datetime.now().strftime("%Y-%m-%d")
    dates = sorted(list(set(dates)))
    if len(dates) == 1: return dates[0]
    return f"{dates[0]}_to_{dates[-1]}"


# -----------------------------
# Initialization & State
# -----------------------------
if "selected_dates" not in st.session_state:
    st.session_state["selected_dates"] = read_dates_from_txt()

current_ui_step = 1
if st.session_state.get("crawling_status") == "running":
    current_ui_step = 2
elif st.session_state.get("crawling_status") == "done":
    current_ui_step = 3
    if st.session_state.get("step3_status") == "running":
        current_ui_step = 3
    elif st.session_state.get("step3_status") == "done":
        current_ui_step = 4
        if st.session_state.get("step4_status") == "running":
            current_ui_step = 4
        elif st.session_state.get("step4_status") == "done":
            current_ui_step = 5

# -----------------------------
# GLOBAL SIDEBAR LAYOUT
# -----------------------------
with st.sidebar:
    st.title("🎛️ Workflow Control")

    # --- STEP 1 ---
    prefix = "🌟 " if current_ui_step == 1 else ""
    st.markdown(f"### {prefix}Step 1: Select Date")

    col_add, col_del = st.columns([1, 1])
    with col_add:
        new_date = st.date_input("Pick Date", key="date_input", label_visibility="collapsed")
        if st.button("➕ Add"):
            date_str = new_date.strftime("%Y-%m-%d")
            if date_str not in st.session_state["selected_dates"]:
                st.session_state["selected_dates"].append(date_str)
                st.session_state["selected_dates"].sort()
                write_dates_to_txt(st.session_state["selected_dates"])
                st.success(f"Added {date_str}")
    with col_del:
        if st.session_state["selected_dates"]:
            del_choice = st.selectbox("Del", st.session_state["selected_dates"], label_visibility="collapsed")
            if st.button("🗑 Del"):
                if del_choice in st.session_state["selected_dates"]:
                    st.session_state["selected_dates"].remove(del_choice)
                    write_dates_to_txt(st.session_state["selected_dates"])
                    st.rerun()

    if st.session_state["selected_dates"]:
        st.caption("Selected: " + ", ".join(st.session_state["selected_dates"]))
    else:
        st.caption("No dates selected.")

    # Check Validity Button
    if st.button("🔎 Check Availability"):
        if not st.session_state["selected_dates"]:
            st.warning("No dates to check.")
        else:
            with st.spinner("Checking dates..."):
                invalid_dates = []
                headers = {'User-Agent': 'Mozilla/5.0'}

                for d in st.session_state["selected_dates"]:
                    # 1. Check for Weekend (Sat=5, Sun=6)
                    try:
                        dt_obj = datetime.strptime(d, "%Y-%m-%d")
                        if dt_obj.weekday() >= 5:
                            invalid_dates.append(f"{d} (Weekend - No papers)")
                            continue
                    except Exception:
                        invalid_dates.append(f"{d} (Invalid Format)")
                        continue

                    # 2. Check URL
                    url = f"https://arxiv.org/catchup/cs.CV/{d}?abs=True"
                    try:
                        r = requests.get(url, headers=headers, timeout=10)
                        if r.status_code != 200:
                            invalid_dates.append(f"{d} (Error: {r.status_code})")
                        elif "No submissions" in r.text or "0 submissions" in r.text:
                            invalid_dates.append(f"{d} (No papers found)")
                    except Exception as e:
                        invalid_dates.append(f"{d} (Connection Error)")

                if invalid_dates:
                    st.error(f"❌ Issues found:\n\n" + "\n".join(invalid_dates))
                else:
                    st.success("✅ All dates are valid weekdays with submissions!")

    st.divider()

    # --- STEP 2 ---
    prefix = "🌟 " if current_ui_step == 2 else ""
    st.markdown(f"### {prefix}Step 2: Crawl")
    if st.button("🚀 Start Crawling", key="btn_crawl"):
        if not st.session_state["selected_dates"]:
            st.error("Select a date first!")
        else:
            
            clean_previous_run()
            st.session_state["crawling_status"] = "running"
            st.session_state["step3_status"] = "ready"
            st.session_state["step4_status"] = "ready"
            st.session_state["step5_done"] = False

    st.divider()

    # --- STEP 3 ---
    prefix = "🌟 " if current_ui_step == 3 else ""
    st.markdown(f"### {prefix}Step 3: Refine (LLM)")

    s3_max = st.slider("Max Items", 5, 100, 20, key="s3_max")
    s3_expand = st.toggle("Expand Details", False, key="s3_expand")

    if st.button("Start Refinement", key="btn_refine", disabled=(current_ui_step < 3)):
        st.session_state["step3_status"] = "running"

    st.divider()

    # --- STEP 4 ---
    prefix = "🌟 " if current_ui_step == 4 else ""
    st.markdown(f"### {prefix}Step 4: Classify")

    if st.button("Start Classification", key="btn_classify", disabled=(current_ui_step < 4)):
        st.session_state["step4_status"] = "running"

    st.divider()

    # --- STEP 5 ---
    prefix = "🌟 " if current_ui_step == 5 else ""
    st.markdown(f"### {prefix}Step 5: Similarity")

    if st.button("Start Similarity", key="btn_sim", disabled=(current_ui_step < 5)):
        st.session_state["step5_running"] = True

# ====================================================================
# MAIN PAGE CONTENT
# ====================================================================

st.title("📘 Web Data Crawling and Analysis: arXiv CS.CV")

# -----------------------------
# Persistent Progress Notices
# -----------------------------
if st.session_state.get("crawling_status") == "done":
    st.success("✅ **Step 1 Completed:** Dates saved in dates.txt")

if st.session_state.get("crawling_status") == "done":
    st.success("✅ **Step 2 Completed:** Crawling finished and raw JSON saved.")

if st.session_state.get("step3_status") == "done":
    st.success("✅ **Step 3 Completed:** LLM Refinement finished.")

if st.session_state.get("step4_status") == "done":
    st.success("✅ **Step 4 Completed:** Paper Classification finished.")

if st.session_state.get("step5_done"):
    st.success("✅ **Step 5 Completed:** Similarity Analysis and Plotly generation finished.")

# -----------------------------
# LOGIC: STEP 2 (Crawling)
# -----------------------------
if st.session_state.get("crawling_status") == "running":
    st.header("Step 2: Crawling from arXiv...")

    selected_dates = st.session_state["selected_dates"]
    st.info(f"Processing dates: {selected_dates}")

    st.session_state["done_global"] = 0
    st.session_state["grand_total"] = 0
    st.session_state["start_global"] = time.time()

    prog = st.progress(0)
    status = st.empty()

    crawl_module = import_crawl_module()

    all_targets = []
    for d in selected_dates:
        url = crawl_module.build_catchup_url("cs.CV", d, with_abs=True)
        try:
            items, meta = crawl_module.crawl_catchup(url)
        except RuntimeError as e:

            st.error(
                "No entries were extracted from the selected catch-up page. "
                "Please check that:\n"
                "1) The chosen date actually has 'New submissions' or 'Cross submissions' "
                "for the selected subject in this dates; and\n"
                "2) The generated URL is a valid arXiv catch-up page (preferably including '?abs=True').\n\n"
                f"Details: {e}"
            )
            st.stop()
        except Exception as e:
            st.error(f"An error occurred during crawling: {e}")
            st.stop()
        all_targets.append((d, url, items, meta))
        st.session_state["grand_total"] += len(items)


    def ui_progress_cb(done_local, total_local, elapsed_local, eta_local, current_date_file):
        done_g = st.session_state["done_global"]
        total_g = st.session_state["grand_total"]
        start_g = st.session_state["start_global"]

        pct = int(done_g / total_g * 100) if total_g else 100
        prog.progress(min(pct, 100))

        elapsed_g = time.time() - start_g
        if elapsed_g > 0 and done_g > 0:
            speed = done_g / elapsed_g
            eta_g = (total_g - done_g) / speed
        else:
            eta_g = 0

        status.text(
            f"Step 2 Crawling: {done_g}/{total_g} ({pct}%) "
            f"| Current file: arxiv_{current_date_file}.json | ETA {int(eta_g)}s"
        )


    for d, url, items, meta in all_targets:
        meta["catchup_date"] = d
        meta["subject"] = "cs.CV"

        status.text(f"Step 2 Parsing: {d}, {len(items)} papers found. Starting details...")


        def local_cb(done_local, total_local, elapsed_local, eta_local):
            st.session_state["done_global"] += 1
            ui_progress_cb(done_local, total_local, elapsed_local, eta_local, d)


        crawl_module.enrich_with_detail(items, max_per_minute=30, progress_cb=local_cb)

        out_json = os.path.join("./crawl_data", f"arxiv_{d}.json")
        crawl_module.export_json(items, meta, out_json, jsonl=False)

    prog.progress(100)
    status.text("Crawling Done")
    st.session_state["crawling_status"] = "done"
    st.rerun()

# -----------------------------
# LOGIC: STEP 3 (Refinement)
# -----------------------------
if st.session_state.get("step3_status") == "running":
    st.header("Step 3: LLM Refinement")
    status_text = st.empty()
    bar = st.progress(0)


    def step3_cb(done, total, elapsed, eta, cur_file):
        pct = int(done / total * 100) if total else 100
        bar.progress(min(pct, 100))
        status_text.text(
            f"Step 3 Refining: {done}/{total} ({pct}%) "
            f"| Current file: {cur_file} | ETA {int(eta)}s"
        )


    try:
        refined_data = data_process(progress_cb=step3_cb)
        bar.progress(100)

        dates_label = build_dates_label_from_session()
        REFINE_FILE = BASE_DIR / "crawl_data" / "refined" / f"arxiv_{dates_label}_refined.json"
        REFINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REFINE_FILE, "w", encoding="utf-8") as f:
            json.dump(refined_data, f, ensure_ascii=False, indent=2)

        st.session_state["refined_file"] = str(REFINE_FILE)
        st.session_state["refined_data"] = refined_data
        st.session_state["step3_status"] = "done"
        st.rerun()
    except Exception as e:
        st.error(f"Step 3 Failed: {e}")
        st.session_state["step3_status"] = "ready"

# DISPLAY STEP 3 RESULTS
if st.session_state.get("step3_status") == "done":
    st.subheader("Step 3 Results: Refined Papers")

    refined_list = st.session_state.get("refined_data", [])
    selected_dates = st.session_state.get("selected_dates", [])

    # 1. Load ALL raw files (containing Author/Title info)
    raw_list = load_all_raw_arxiv_data(selected_dates)

    # 2. Create a Lookup Map based on 'arxiv_id' (Unique Key)
    raw_map = {}
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, dict):
                # Use arxiv_id as key if available
                aid = item.get("arxiv_id")
                if aid:
                    raw_map[aid] = item

    # 3. Merge Logic: Iterate Refined, Pull Info from Raw Map
    merged_results = []
    for item in refined_list:
        if not isinstance(item, dict): continue

        aid = item.get("arxiv_id")
        # Get raw info using ID, or empty dict if not found
        raw_info = raw_map.get(aid, {})

        # Start with Refined item (has summary/links), update with Raw (has authors/title)
        merged = item.copy()

        # If raw data exists, verify we overwrite title/authors with the clean version
        if raw_info:
            merged["title"] = raw_info.get("title", item.get("title"))
            merged["authors_listing"] = raw_info.get("authors_listing", [])
            # If raw abstract exists, keep it too
            merged["raw_abstract"] = raw_info.get("abstract", "")

        merged_results.append(merged)

    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Total Papers", len(merged_results))
    max_show = st.session_state.get("s3_max", 20)
    m2.metric("Showing Top", min(len(merged_results), max_show))

    st.divider()

    # --- CLEAN SIMPLE LIST ---
    is_expanded = st.session_state.get("s3_expand", False)

    for idx, item in enumerate(merged_results[:max_show], start=1):

        # Extract Data
        title = item.get("title", "No Title Found")
        concise = item.get("concise_description", "No AI summary available.")

        # Try getting abstract from Raw (clean), then Refined 'original_text'
        orig = item.get("raw_abstract", item.get("original_text", ""))
        # If original text is the mashup "no.1 title...", we might leave it or try to clean it,
        # but usually raw_abstract is present if Step 2 ran correctly.

        # Authors (from authors_listing in Raw)
        authors_list = item.get("authors_listing", [])
        if authors_list:
            auth_str = ", ".join(authors_list)
        else:
            auth_str = "Unknown"

        # Links & Metadata
        link = item.get("html_url", "#")
        cat = item.get("section", "cs.CV")
        aid = item.get("arxiv_id", "N/A")

        # Fallback Parsing (If Raw Map failed completely)
        if title == "No Title Found" and "original_text" in item:
            # Try to parse "no.1 title:Actual Title abstract:..."
            match = re.search(r"title:(.*?)abstract:", item["original_text"], re.IGNORECASE)
            if match:
                title = match.group(1).strip()

        # UI: Simple Expander
        with st.expander(f"📄 {idx}. {title}", expanded=is_expanded):
            st.markdown(f"**👨‍🔬 Authors:** {auth_str}")
            st.markdown(f"**🔗 Link:** [{aid}]({link}) | **Category:** {cat}")

            st.divider()
            st.markdown("**📝 AI Summary:**")
            st.info(concise)

            st.markdown("**📄 Original Abstract:**")
            st.caption(orig)

# -----------------------------
# LOGIC: STEP 4 (Classification)
# -----------------------------
if st.session_state.get("step4_status") == "running":
    st.header("Step 4: Classification")
    status_text = st.empty()
    bar = st.progress(0)

    try:
        def step4_cb(done, total, elapsed, eta, cur_file):
            pct = int(done / total * 100) if total else 100
            bar.progress(min(pct, 100))
            status_text.text(
                f"Step 4 Classifying: {done}/{total} ({pct}%) "
                f"| Current file: {cur_file} | ETA {int(eta)}s"
            )


        refined_data = st.session_state.get("refined_data")
        refined_file = st.session_state.get("refined_file")

        dates_label = build_dates_label_from_session()
        REFINE_FILE = BASE_DIR / "crawl_data" / "refined" / f"arxiv_{dates_label}_refined.json"

        if isinstance(refined_data, list):
            REFINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(REFINE_FILE, "w", encoding="utf-8") as f:
                json.dump(refined_data, f, ensure_ascii=False, indent=2)
            refined_input_path = str(REFINE_FILE)
        else:
            refined_input_path = refined_file if refined_file else str(REFINE_FILE)

        classified_data = classify(input_path=refined_input_path, progress_cb=step4_cb, paper_progress=True)

        CLASSIFIED_FILE = BASE_DIR / "crawl_data" / "classified" / f"arxiv_{dates_label}_classified.json"
        CLASSIFIED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CLASSIFIED_FILE, "w", encoding="utf-8") as f:
            json.dump(classified_data, f, ensure_ascii=False, indent=2)

        st.session_state["classified_file"] = str(CLASSIFIED_FILE)
        st.session_state["step4_status"] = "done"
        bar.progress(100)
        st.rerun()

    except Exception as e:
        st.error(f"Step 4 Error: {e}")
        st.session_state["step4_status"] = "ready"

if st.session_state.get("step4_status") == "done":
    st.subheader("Step 4 Results: Classification Report")
    c_file = st.session_state.get("classified_file")
    if c_file and os.path.exists(c_file):
        with open(c_file, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        raw_text = ""
        if isinstance(c_data, list) and c_data:
            raw_text = c_data[0].get("raw_classification_text", "")
        elif isinstance(c_data, dict):
            raw_text = c_data.get("raw_classification_text", "")

        st.text_area("Overall Classification", value=raw_text, height=300)

# -----------------------------
# LOGIC: STEP 5 (Similarity)
# -----------------------------
if st.session_state.get("step5_running"):
    st.header("Step 5: Similarity Calculation")

    # Replaced progress bar with spinner
    with st.spinner("Calculating similarity matrix and generating plots... This may take a few moments."):
        sim_result = similarity_calculate()

        st.session_state["sim_result"] = sim_result
        st.session_state["step5_done"] = True
        st.session_state["step5_running"] = False
        st.rerun()

if st.session_state.get("step5_done"):
    sim_result = st.session_state.get("sim_result", {})

    st.markdown("---")
    st.header("Step 5: Interactive Cluster Map")

    if sim_result:
        base_name = list(sim_result.keys())[0]
        html_path = sim_result[base_name].get("html_path")

        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=800,width=1200, scrolling=True)
        else:
            fig = sim_result[base_name].get("plotly_fig")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No plot data available.")

    # PNG Viewer Section
    st.markdown("---")
    st.subheader("Static Similarity Images")

    if "show_png_viewer" not in st.session_state:
        st.session_state["show_png_viewer"] = False

    if st.button("Load Static Images"):
        loader = ImageLoader(IMG_DIR)
        pngs = loader.list_pngs()
        if pngs:
            st.session_state["png_list"] = pngs
            st.session_state["show_png_viewer"] = True
        else:
            st.warning("No PNGs found.")

    if st.session_state.get("show_png_viewer") and st.session_state.get("png_list"):
        pngs = st.session_state["png_list"]
        total = len(pngs)
        if "png_idx" not in st.session_state: st.session_state["png_idx"] = 0

        idx = st.session_state["png_idx"] % total
        cur_path = pngs[idx]

        st.image(str(cur_path), caption=f"{idx + 1}/{total}: {cur_path.name}", use_column_width=True)

        c1, c2 = st.columns(2)
        if c1.button("⬅️ Prev"):
            st.session_state["png_idx"] -= 1
            st.rerun()
        if c2.button("Next ➡️"):
            st.session_state["png_idx"] += 1
            st.rerun()