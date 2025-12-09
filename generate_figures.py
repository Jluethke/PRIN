import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import fitz  # PyMuPDF for PDF reading

# -----------------------------------------------------------
# PATHS
# -----------------------------------------------------------

FIG_DIR = r"G:\othreflashdrive\projects2\PRIN-master\PRIN-master\empirical\figures"
TABLE_DIR = r"G:\othreflashdrive\projects2\PRIN-master\PRIN-master\empirical\tables"

OUT_COMPOSITES = r"G:\othreflashdrive\projects2\PRIN-master\PRIN-master\empirical\composites"
OUT_TABLES = r"G:\othreflashdrive\projects2\PRIN-master\PRIN-master\empirical\table_images"

os.makedirs(OUT_COMPOSITES, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

# -----------------------------------------------------------
# Utility: Load image (PDF or PNG/JPG)
# -----------------------------------------------------------
def load_image(path):
    if path.endswith(".pdf"):
        # Load first page via PyMuPDF
        doc = fitz.open(path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        doc.close()
        return img.convert("RGB")
    return Image.open(path).convert("RGB")

# -----------------------------------------------------------
# Utility: Create a 4-panel composite
# -----------------------------------------------------------
def create_composite(output_name, image_names, titles=None):
    images = []
    for name in image_names:
        path = os.path.join(FIG_DIR, name)
        images.append(load_image(path))

    target_width = 800
    resized = []
    for img in images:
        w, h = img.size
        scale = target_width / w
        resized.append(img.resize((target_width, int(h * scale))))

    w, h = resized[0].size
    composite = Image.new("RGB", (w * 2, h * 2), "white")

    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, pos in zip(resized, positions):
        composite.paste(img, pos)

    out_path = os.path.join(OUT_COMPOSITES, output_name)
    composite.save(out_path, dpi=(300, 300))
    print(f"[SAVED] Composite: {out_path}")

# -----------------------------------------------------------
# Utility: Convert LaTeX table (.tex) into an image
# -----------------------------------------------------------
def tex_table_to_image(tex_file, output_name, title=None):
    path = os.path.join(TABLE_DIR, tex_file)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the tabular environment
    tabular = re.findall(r"\\begin{tabular}{.*?}(.*?)\\end{tabular}", content, re.S)
    if not tabular:
        print(f"[ERROR] No tabular found in {tex_file}")
        return

    block = tabular[0]

    # Split into raw lines
    raw_lines = block.split("\\\\")
    rows = []

    for line in raw_lines:
        line = line.strip()

        # Remove formatting commands
        if not line:
            continue
        if any(cmd in line for cmd in ["\\hline", "\\toprule", "\\midrule", "\\bottomrule"]):
            continue

        # Try splitting on ampersand first
        if "&" in line:
            parts = [p.strip() for p in line.split("&")]
        else:
            # Split on 2+ spaces
            parts = re.split(r"\s{2,}", line)
            parts = [p.strip() for p in parts if p.strip()]

        if parts:
            rows.append(parts)

    if not rows:
        print(f"[ERROR] No valid rows extracted: {tex_file}")
        return

    # Determine header
    # If first row contains no letters → generate headers
    if all(not any(c.isalpha() for c in cell) for cell in rows[0]):
        header = [f"Col{j+1}" for j in range(len(rows[0]))]
        data = rows
    else:
        header = rows[0]
        data = rows[1:]

    # Build DataFrame
    df = pd.DataFrame(data, columns=header)

    # Create figure
    fig_width = len(df.columns) * 2.2
    fig_height = len(df) * 0.6 + 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title:
        plt.title(title, fontsize=16, pad=20)

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc="center",
                     loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    out_path = os.path.join(OUT_TABLES, output_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Table image: {out_path}")




# -----------------------------------------------------------
# CREATE ALL COMPOSITES
# -----------------------------------------------------------

create_composite("Figure_3_1_Chaos.png", [
    "chaos_map.pdf",
    "chaos_gate.pdf",
    "chaos_contraction_effect.pdf",
    "chaos_variance_reduction.pdf"
])

create_composite("Figure_4_1_PredictiveCoding.png", [
    "predictive_coding_contraction.pdf",
    "predictive_coding_error_trajectory.pdf",
    "jacobian_norm.pdf",
    "spectral_radius.pdf"
])

create_composite("Figure_4_2_Resonance.png", [
    "resonance_strength.pdf",
    "resonance_gate.pdf",
    "resonance_crosscorr.pdf",
    "spectral_heatmap.pdf"
])

create_composite("Figure_4_3_MultiHead.png", [
    "spectral_radius.pdf",
    "spectral_heatmap.pdf",
    "resonance_crosscorr.pdf",
    "hidden_state_norm.pdf"
])

# -----------------------------------------------------------
# CONVERT TABLES
# -----------------------------------------------------------

tex_table_to_image("mcs_results.tex", "MCS_Results.png", "Model Confidence Set")
tex_table_to_image("stat_tests.tex", "StatTests.png", "Statistical Significance Tests")
tex_table_to_image("rolling_dm_regimes.tex", "RollingDM.png", "Rolling Diebold–Mariano Test")

print("\nALL DONE — Figures and tables generated successfully!")
