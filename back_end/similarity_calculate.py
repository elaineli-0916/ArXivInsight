import json
import re
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 0. Basic paths and model configuration ---
REFINED_DIR = "./crawl_data/refined"
CLASS_DIR = "./crawl_data/refined_output"
OUTPUT_ROOT = "./crawl_data/similarity_results"
MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# PCA helper
# ============================================================
def _to_2d_coords(emb_np: np.ndarray) -> np.ndarray:
    """
    Reduce embeddings to 2D (PC1, PC2).
    """
    if emb_np.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_np)
    else:
        coords = emb_np
    return coords


# ============================================================
# 1. Parse classification result file
# ============================================================
def parse_classification_file(filepath):
    """
    Parse a classification text file and return
    a list of (category_name, [id list]).
    """
    categories = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = re.compile(r"Category: (.*?)\nIDs: ([\d, ]+)", re.DOTALL)
        matches = pattern.findall(content)
        for name, ids_str in matches:
            category_name = name.strip()
            id_list = [
                int(id_str.strip())
                for id_str in ids_str.split(",")
                if id_str.strip()
            ]
            categories.append((category_name, id_list))
    except FileNotFoundError:
        print(f"Error: file '{filepath}' not found.")
        return None
    return categories


# ============================================================
# 2. Load refined JSON  --->  id -> meta (arxiv_id, section, html_url)
# ============================================================
def load_papers_data(filepath):
    """
    Load refined JSON and build a mapping:
        id -> {
            "arxiv_id": str,
            "section": str,
            "html_url": str,
        }
    """
    id_to_meta = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            papers = json.load(f)
        for paper in papers:
            pid = paper.get("id")
            if pid is None:
                continue
            id_to_meta[pid] = {
                "arxiv_id": paper.get("arxiv_id", ""),
                "section": paper.get("section", ""),
                "html_url": paper.get("html_url", ""),
            }
    except FileNotFoundError:
        print(f"Error: file '{filepath}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: file '{filepath}' is not valid JSON.")
        return None
    return id_to_meta


# ============================================================
# 3. Sanitize filename
# ============================================================
def sanitize_filename(name):
    """
    Convert arbitrary category names to safe filenames.
    """
    name = re.sub(r"[\s/]", "_", name)
    name = re.sub(r'[\\:*?"<>|]', "", name)
    return name[:100] + ".txt"


# ============================================================
# 4. Scan refined + classification files
# ============================================================
def collect_file_pairs():
    """
    Scan REFINED_DIR and CLASS_DIR and return:
    [(base_name, json_path, class_path), ...]
    """
    pairs = []
    if not os.path.isdir(REFINED_DIR):
        print(f"Error: directory '{REFINED_DIR}' does not exist.")
        return pairs

    for fname in os.listdir(REFINED_DIR):
        if not fname.endswith("_refined.json"):
            continue
        base_name = fname[:-5]
        json_path = os.path.join(REFINED_DIR, fname)
        class_fname = base_name + "_classification.txt"
        class_path = os.path.join(CLASS_DIR, class_fname)
        if os.path.exists(class_path):
            pairs.append((base_name, json_path, class_path))
        else:
            print(
                f"[Warning] Found JSON: {json_path}, "
                f"but classification file is missing: {class_path}"
            )
    return pairs


# ============================================================
# 5. Per-category Matplotlib scatter (PNG)
# ============================================================
def plot_category_cluster(
    base_name,
    category_name,
    ids_in_order,
    embeddings,
    similarity_results,
    output_dir,
):
    """
    Draw a small 2D cluster for a single category using Matplotlib.
    """
    if len(ids_in_order) < 2:
        return

    emb_np = embeddings.detach().cpu().numpy()

    if emb_np.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_np)
    else:
        coords = emb_np

    top_pair = similarity_results[0]
    id_a, id_b, top_sim = top_pair["id1"], top_pair["id2"], top_pair["similarity"]

    try:
        idx_a = ids_in_order.index(id_a)
        idx_b = ids_in_order.index(id_b)
    except ValueError:
        idx_a = idx_b = None

    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)

    if idx_a is not None and idx_b is not None:
        plt.scatter(coords[idx_a, 0], coords[idx_a, 1])
        plt.scatter(coords[idx_b, 0], coords[idx_b, 1])
        plt.annotate(
            str(id_a),
            (coords[idx_a, 0], coords[idx_a, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
        plt.annotate(
            str(id_b),
            (coords[idx_b, 0], coords[idx_b, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.title(
        f"{base_name}\n{category_name}\n"
        f"Closest pair: ({id_a}, {id_b}), sim={top_sim:.3f}"
    )
    plt.tight_layout()

    png_name = sanitize_filename(category_name).replace(".txt", "_cluster.png")
    png_path = os.path.join(output_dir, png_name)
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"      Category cluster PNG saved to: {png_path}")


# ============================================================
# 6. Global Matplotlib cluster (PNG)
# ============================================================
def plot_big_cluster(
    base_name,
    all_ids,
    all_embeddings,
    all_cat_names,
    top_pair_by_cat,
    output_dir,
):
    """
    Draw a global 2D cluster for all categories using Matplotlib.
    """
    if len(all_ids) < 2:
        return

    emb_np = np.vstack(all_embeddings)
    if emb_np.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_np)
    else:
        coords = emb_np

    unique_cats = sorted(list(set(all_cat_names)))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    labels = np.array([cat_to_idx[c] for c in all_cat_names])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, alpha=0.7)

    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_cats, fontsize=6, loc="best")

    id_to_pos = {pid: i for i, pid in enumerate(all_ids)}

    # highlight closest pairs per category
    for cat, pair in top_pair_by_cat.items():
        id1, id2, sim = pair["id1"], pair["id2"], pair["similarity"]
        if id1 not in id_to_pos or id2 not in id_to_pos:
            continue
        i = id_to_pos[id1]
        j = id_to_pos[id2]
        x1, y1 = coords[i]
        x2, y2 = coords[j]

        plt.plot([x1, x2], [y1, y2])
        plt.annotate(
            str(id1),
            (x1, y1),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )
        plt.annotate(
            str(id2),
            (x2, y2),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )

    plt.title(f"{base_name} - All Categories (PCA 2D)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "all_categories_big_cluster.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"    ★ Global cluster PNG saved to: {out_path}")


# ============================================================
# 7. Plotly global cluster (interactive, short text + click to URL)
# ============================================================
def plot_big_cluster_plotly(
    base_name,
    all_ids,
    all_embeddings,
    all_cat_names,
    top_pair_by_cat,
    id_meta_map,
    output_dir=None,
    save_html=True,
):
    """
    Plot an interactive Plotly cluster figure.

    Hover text contains only:
        - ID
        - arxiv_id
        - section

    When clicking on a point in the exported HTML, the browser
    will open the corresponding html_url in a new tab.
    """
    if len(all_ids) < 2:
        return None

    emb_np = np.asarray(all_embeddings)
    coords = _to_2d_coords(emb_np)

    arxiv_ids = [
        (id_meta_map.get(pid, {}).get("arxiv_id", "") or "")
        for pid in all_ids
    ]
    sections = [
        (id_meta_map.get(pid, {}).get("section", "") or "")
        for pid in all_ids
    ]
    urls = [
        (id_meta_map.get(pid, {}).get("html_url", "") or "")
        for pid in all_ids
    ]

    df = pd.DataFrame(
        {
            "id": all_ids,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "category": all_cat_names,
            "arxiv_id": arxiv_ids,
            "section": sections,
            "url": urls,
        }
    )

    fig = go.Figure()

    # --- scatter points by category ---
        # --- scatter points by category ---
    for cat_name, sub in df.groupby("category"):
        customdata = np.column_stack(
            (
                sub["arxiv_id"].astype(str),
                sub["section"].astype(str),
                sub["url"].astype(str),
                sub["id"].astype(str),
                sub["category"].astype(str),  # <- customdata[4]
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sub["pc1"],
                y=sub["pc2"],
                mode="markers",
                name=str(cat_name),
                customdata=customdata,
                hovertemplate=(
                    "ID: %{customdata[3]}<br>"
                    "arXiv ID: %{customdata[0]}<br>"
                    "Section: %{customdata[1]}<br>"
                    "Category: %{customdata[4]}<extra></extra>"
                ),
                marker=dict(size=7, opacity=1),
                hoverlabel=dict(font_size=12), 
            )
        )


    # --- build a coordinate map for highlighting pairs ---
    coord_map = {
        int(row["id"]): (row["pc1"], row["pc2"])
        for _, row in df.iterrows()
    }

    # --- highlight closest pairs ---
    for cat, pair in top_pair_by_cat.items():
        id1, id2, sim = int(pair["id1"]), int(pair["id2"]), pair["similarity"]

        if id1 not in coord_map or id2 not in coord_map:
            continue

        x1, y1 = coord_map[id1]
        x2, y2 = coord_map[id2]

        # line between the pair
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                showlegend=False,
                line=dict(width=2),
                hoverinfo="skip",
            )
        )

        # endpoints with the same short hover info
        meta1 = id_meta_map.get(id1, {})
        meta2 = id_meta_map.get(id2, {})
        customdata_pair = np.array(
            [
                [
                    meta1.get("arxiv_id", ""),
                    meta1.get("section", ""),
                    meta1.get("html_url", ""),
                    str(id1),
                    cat,
                ],
                [
                    meta2.get("arxiv_id", ""),
                    meta2.get("section", ""),
                    meta2.get("html_url", ""),
                    str(id2),
                    cat,
                ],
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="markers+text",
                text=[str(id1), str(id2)],
                textposition="top center",
                showlegend=False,
                customdata=customdata_pair,
                hovertemplate=(
                    "ID: %{customdata[3]}<br>"
                    "arXiv ID: %{customdata[0]}<br>"
                    "Section: %{customdata[1]}<br>"
                    "Category: %{customdata[4]}<extra></extra>"
                ),
                marker=dict(size=12, symbol="circle-open"),
                hoverlabel=dict(font_size=12),  
            )
        )

    fig.update_layout(
        title=f"{base_name} - Plotly PCA 2D Cluster",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        height=700,
        width=900,
        legend_title="Category",
        clickmode="event",
    )
    html_path = None  
    if save_html and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        div_id = f"{base_name}_plot_div"
        html_path = os.path.join(output_dir, f"{base_name}_plotly_cluster.html")

        # JS snippet: click to open url from customdata[2]
        post_script =f"""
var gd = document.getElementById('{div_id}');
if (gd) {{
    var infoId = '{div_id}_info';
    var infoDiv = document.getElementById(infoId);
    if (!infoDiv) {{
        infoDiv = document.createElement('div');
        infoDiv.id = infoId;
        infoDiv.style.marginTop = '8px';
        infoDiv.style.fontFamily = 'sans-serif';
        infoDiv.style.fontSize = '14px';
        infoDiv.textContent = 'Click a point to show its category.';
        gd.parentNode.insertBefore(infoDiv, gd.nextSibling);
    }}

    gd.on('plotly_click', function(data) {{
        if (!data || !data.points || !data.points.length) return;
        var pt = data.points[0];
        if (!pt.customdata || pt.customdata.length < 3) return;

        var url = pt.customdata[2];   // html_url
        var category = pt.customdata[4] || '';  
        infoDiv.textContent = 'Category: ' + category;
        if (url) {{
            window.open(url, '_blank');
        }}
    }});
}}
"""

        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
            full_html=True,
            div_id=div_id,
            post_script=post_script,
        )
        print(f"    ★ Plotly interactive cluster saved to: {html_path}")

    return fig, html_path  


# ============================================================
# 8. Main entry (PNG + Plotly)
# ============================================================
def similarity_calculate():
    """
    Run similarity analysis over all matched refined + classification files.

    Returns:
        dict:
        {
            base_name: {
                "plotly_fig": fig,
                "output_dir": str,
                "top_pair_by_cat": dict
            },
            ...
        }
    """
    print("Scanning refined / refined_output and pairing files...")
    file_pairs = collect_file_pairs()
    if not file_pairs:
        print("No valid file pairs found.")
        return {}

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    result = {}

    print("\n================ Start processing all file pairs ================\n")

    for base_name, json_path, class_path in file_pairs:
        print(f">>> Processing dataset: {base_name}")
        print(f"    Refined JSON: {json_path}")
        print(f"    Classification file: {class_path}")

        all_categories = parse_classification_file(class_path)
        if not all_categories:
            print(f"    [Skip] Failed to parse classification file: {class_path}")
            continue

        id_meta_map = load_papers_data(json_path)
        if not id_meta_map:
            print(f"    [Skip] Failed to load JSON file: {json_path}")
            continue

        output_dir = os.path.join(OUTPUT_ROOT, base_name)
        os.makedirs(output_dir, exist_ok=True)

        all_ids_global = []
        all_embs_global = []
        all_cats_global = []
        top_pair_by_cat = {}

        # --------------------------------------------------------------
        # Per-category similarity and small plots
        # --------------------------------------------------------------
        for category_name, category_ids in all_categories:
            print(f"    - Category: {category_name}")

            descriptions_to_encode = []
            ids_in_order = []

            for pid in category_ids:
                meta = id_meta_map.get(pid)
                # use arxiv_id / section only for display;
                # embeddings still use concise info? Here we don't have it anymore,
                # so we can simply skip if we don't have html_url etc.
                # For similarity, you probably already used concise_description;
                # if you still need that, extend id_meta_map to include it.
                # Here, we just require meta to exist.
                if meta:
                    # For embeddings you may want concise_description;
                    # replace the line below with paper["concise_description"]
                    # if you keep it in JSON.
                    text = meta.get("arxiv_id") or str(meta.get("html_url") or pid)
                    descriptions_to_encode.append(text)
                    ids_in_order.append(pid)
                else:
                    print(
                        f"      [Warning] ID={pid} not found in refined JSON, skipped."
                    )

            if len(descriptions_to_encode) < 2:
                print("      [Skip] Fewer than 2 valid papers in this category.")
                continue

            embeddings = model.encode(
                descriptions_to_encode, convert_to_tensor=True
            )

            # compute pairwise cosine similarity
            similarity_results = []
            for i, j in combinations(range(len(ids_in_order)), 2):
                score = util.cos_sim(embeddings[i], embeddings[j]).item()
                similarity_results.append(
                    {
                        "id1": ids_in_order[i],
                        "id2": ids_in_order[j],
                        "similarity": score,
                    }
                )
            similarity_results.sort(
                key=lambda x: x["similarity"], reverse=True
            )

            # write text result
            out_name = sanitize_filename(category_name)
            text_out_path = os.path.join(output_dir, out_name)
            with open(text_out_path, "w", encoding="utf-8") as f:
                f.write("Similarity results\n")
                f.write(f"Dataset: {base_name}\n")
                f.write(f"Category: {category_name}\n")
                f.write("=" * 50 + "\n\n")
                for r in similarity_results:
                    f.write(
                        f"(ID {r['id1']}, ID {r['id2']}): "
                        f"{r['similarity']:.4f}\n"
                    )
            print(f"      Text similarity result saved to: {text_out_path}")

            # per-category small PNG
            plot_category_cluster(
                base_name,
                category_name,
                ids_in_order,
                embeddings,
                similarity_results,
                output_dir,
            )

            emb_np = embeddings.detach().cpu().numpy()
            all_embs_global.append(emb_np)
            all_ids_global.extend(ids_in_order)
            all_cats_global.extend([category_name] * len(ids_in_order))

            top_pair_by_cat[category_name] = similarity_results[0]

        # --------------------------------------------------------------
        # Global plots: Matplotlib + Plotly
        # --------------------------------------------------------------
        if all_embs_global:
            all_embs_global = np.vstack(all_embs_global)

            # 1) Matplotlib (PNG)
            plot_big_cluster(
                base_name,
                all_ids_global,
                all_embs_global,
                all_cats_global,
                top_pair_by_cat,
                output_dir,
            )

            # 2) Plotly (interactive, short hover + click to URL)
            fig,html_path  = plot_big_cluster_plotly(
                base_name,
                all_ids_global,
                all_embs_global,
                all_cats_global,
                top_pair_by_cat,
                id_meta_map,
                output_dir,
                save_html=True,
            )

            result[base_name] = {
                "plotly_fig": fig,
                "output_dir": output_dir,
                "html_path": html_path,
                "top_pair_by_cat": top_pair_by_cat,
            }

        print(f"<<< Dataset {base_name} finished.\n")

    print("================ All file pairs processed. ================")
    return result
