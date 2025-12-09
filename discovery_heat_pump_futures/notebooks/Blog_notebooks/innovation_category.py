#!/usr/bin/env python3
"""
Heat Pump Innovation Analysis - Plot 2: Innovation Category Distribution

This script creates a horizontal bar chart showing the distribution of innovation focus areas.
Filters documents based on taxonomy-defined subcategories.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
import ast
import warnings

warnings.filterwarnings("ignore")

# Define color palette
COLORS = {
    "primary": ["#182A48", "#18A48C", "#CFECE7", "#D2C9C0"],
    "secondary": ["#97D9E3", "#A59BEE", "#DCF2F5", "#EBE9FB"],
    "neutral": ["#DCDEE3", "#F3EFEB", "#F0F0F0", "#E0E0E0"],
    "others": ["#F2B950", "#0000F5", "#EAA8B7", "#EB003B", "#FDDA99", "#B7B0A9"],
}

# TAXONOMY ADDITION: Define valid subcategories for each category
TAXONOMY = {
    "trad_components": [
        "Compressors",
        "Refrigerants",
        "Heat-exchangers",
        "Motor & Drives",
        "Lubrication & oil management",
    ],
    "non_trad_technologies": [
        "Elastocaloric",
        "Electrocaloric",
        "Magnetocaloric",
        "Ionocaloric",
        "Barocaloric",
        "Thermoelectric",
        "Electro & Chemisorption",
        "Thermoacoustic",
        "Sorption / Absorption",
        "Hybrid & cascade",
    ],
    "system_design": [
        "Commissioning & installation",
        "Controls & optimisation",
        "Operational integration",
    ],
    "other_improvements": [
        "Flexible cycles",
        "Defrost & icing mitigation",
        "Thermal storage",
    ],
    "circular_economy": [
        "Design-for-disassembly",
        "Modular assemblies",
        "Design-for-recycling & material homogeneity",
        "Repairability & predictive maintenance",
        "Additive manufacturing",
        "Reverse logistics & take-back",
        "Industrial symbiosis / resource sharing",
        "Refrigerant recovery & reclamation",
        "Other circularity principles",
    ],
}

# File paths
ABSOLUTE_PATH = "/home/pascualdiego/projects/DiscoveryHP/inputs/heat_pump_data_w_categories.csv"
RELATIVE_PATH = "inputs/heat_pump_data_w_categories.csv"

# Try to use the absolute path first, fall back to relative if needed
try:
    df = pd.read_csv(ABSOLUTE_PATH)
    print(f"Successfully loaded data from: {ABSOLUTE_PATH}")
except FileNotFoundError:
    try:
        df = pd.read_csv(RELATIVE_PATH)
        print(f"Successfully loaded data from: {RELATIVE_PATH}")
    except FileNotFoundError:
        print("ERROR: Could not find the data file at either path!")
        raise

print("\n" + "=" * 60)
print("STEP 1: DATA STRUCTURE OVERVIEW")
print("=" * 60)

# Display basic information
print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Focus on innovation category columns
category_columns = [
    "trad_components",
    "non_trad_technologies",
    "system_design",
    "other_improvements",
    "circular_economy",
]

print(f"\nInnovation category columns:")
for col in category_columns:
    if col in df.columns:
        print(f" ✓ {col}")
    else:
        print(f" ✗ {col} (NOT FOUND)")

# Display first 10 rows of these columns
print("\nFirst 10 rows of innovation categories:")
if all(col in df.columns for col in category_columns):
    print(df[category_columns].head(10).to_string())
else:
    print("Some category columns are missing!")

# Check for other potential category columns
all_columns = df.columns.tolist()
other_categories = [
    col
    for col in all_columns
    if col not in category_columns
    and col
    not in [
        "id",
        "source",
        "title_abstract",
        "publication_year",
        "is_heat_pump",
        "confidence",
        "reason",
        "application_type",
        "heat_pump_type",
        "specific_applications",
        "technology_readiness_level",
        "trl_reasoning",
    ]
]

if other_categories:
    print(f"\nOther potential category columns found: {other_categories}")

print("\n" + "=" * 60)
print("STEP 2: IDENTIFY EMPTY VALUES AND DATA TYPES")
print("=" * 60)

# Check data types for category columns
print("\nData types for innovation category columns:")
for col in category_columns:
    if col in df.columns:
        print(f" {col}: {df[col].dtype}")

# Check for missing values
print("\nMissing values in innovation category columns:")
for col in category_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df) * 100)
        print(f" {col}: {missing_count:,} ({missing_pct:.1f}%)")

# Sample values to understand format
print("\nSample values from each category (first non-null value):")
for col in category_columns:
    if col in df.columns:
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            print(f" {col}: {non_null_values.iloc[0]}")

print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING WITH TAXONOMY FILTER")
print("=" * 60)

# Create a copy for cleaning
df_clean = df.copy()


# Function to check if a value represents an empty list or is actually empty
def is_empty_category(value):
    if pd.isna(value):
        return True
    if value == "[]":
        return True
    if value == "":
        return True
    return False


# TAXONOMY ADDITION: Function to check if subcategories are valid
def has_valid_subcategories(value, category):
    """Check if the category value contains only valid subcategories from taxonomy"""
    if is_empty_category(value):
        return False

    try:
        # Parse the list
        if isinstance(value, str) and value.startswith("["):
            subcategories = ast.literal_eval(value)
        else:
            subcategories = [str(value).strip()]

        # Check if all subcategories are in the taxonomy
        valid_subcats = TAXONOMY.get(category, [])
        for subcat in subcategories:
            if isinstance(subcat, str):
                # Clean the subcategory name
                clean_subcat = subcat.strip().strip('"').strip("'")
                # Check if it's in the valid list
                if clean_subcat not in valid_subcats:
                    return False
        return True
    except Exception:
        return False


# Track excluded documents per category
excluded_documents_by_category = {cat: [] for cat in category_columns}
excluded_counts = {cat: 0 for cat in category_columns}

documents_with_no_categories = []
documents_with_valid_categories = set()

# Count documents with each innovation category (only valid subcategories)
category_counts = {}
category_counts_total = {}  # Total including invalid subcategories

for col in category_columns:
    if col in df_clean.columns:
        # Count all non-empty entries (for comparison)
        non_empty_mask = ~df_clean[col].apply(is_empty_category)
        total_count = non_empty_mask.sum()
        category_counts_total[col] = total_count

        # Count only entries with valid subcategories
        valid_count = 0
        for idx, row in df_clean.iterrows():
            if not is_empty_category(row[col]):
                if has_valid_subcategories(row[col], col):
                    valid_count += 1
                    documents_with_valid_categories.add(idx)
                else:
                    # Document has category but invalid subcategories
                    excluded_counts[col] += 1
                    excluded_documents_by_category[col].append(
                        row["id"] if "id" in row else idx
                    )

        category_counts[col] = valid_count

        # Get percentages
        percentage_valid = (valid_count / len(df_clean)) * 100
        percentage_total = (total_count / len(df_clean)) * 100

        print(f"\n{col}:")
        print(
            f" Documents with this category (total): {total_count:,} ({percentage_total:.1f}%)"
        )
        print(
            f" Documents with valid subcategories: {valid_count:,} ({percentage_valid:.1f}%)"
        )
        print(
            f" Documents excluded (invalid subcategories): {excluded_counts[col]:,}"
        )

# Check for documents with no categories at all
for idx, row in df_clean.iterrows():
    has_any_category = False
    for col in category_columns:
        if col in df_clean.columns and not is_empty_category(row[col]):
            has_any_category = True
            break
    if not has_any_category:
        documents_with_no_categories.append(row["id"] if "id" in row else idx)

# Clean category names for display
display_names = {
    "trad_components": "Traditional\nComponents",
    "non_trad_technologies": "Non-traditional\nTechnologies",
    "system_design": "System\nDesign",
    "other_improvements": "Other\nImprovements",
    "circular_economy": "Circular\nEconomy",
}

print("\n" + "=" * 60)
print("STEP 4: CLEANING SUMMARY")
print("=" * 60)

print(f"\nTotal documents analyzed: {len(df_clean):,}")
print(
    f"Documents with valid innovation categories: {len(documents_with_valid_categories):,}"
)
print(f"Documents with no categories: {len(documents_with_no_categories):,}")

# ADDITION: Calculate total category instances
total_category_instances = sum(category_counts.values())
print(
    f"Total category instances (sum of all valid categories): {total_category_instances:,}"
)

print("\nDocuments per innovation category (valid subcategories only):")
for col, count in category_counts.items():
    display_name = display_names.get(col, col).replace("\n", " ")
    percentage = (count / len(df_clean)) * 100
    instance_percentage = (count / total_category_instances) * 100
    print(
        f" {display_name}: {count:,} ({percentage:.1f}% of documents, {instance_percentage:.1f}% of instances)"
    )

# Check for documents with multiple categories
multi_category_analysis = []
for idx, row in df_clean.iterrows():
    categories_present = 0
    for col in category_columns:
        if (
            col in df_clean.columns
            and not is_empty_category(row[col])
            and has_valid_subcategories(row[col], col)
        ):
            categories_present += 1
    multi_category_analysis.append(categories_present)

multi_category_counts = (
    pd.Series(multi_category_analysis).value_counts().sort_index()
)

print("\nDocuments by number of valid innovation categories:")
for num_cats, count in multi_category_counts.items():
    print(f" {num_cats} categories: {count:,} documents")

# Count documents with other categories
other_category_counts = {}
if other_categories:
    for cat in other_categories:
        if cat in df.columns:
            count = (~df[cat].apply(is_empty_category)).sum()
            if count > 0:
                other_category_counts[cat] = count

print("\n" + "=" * 60)
print("STEP 5: CREATE VISUALIZATION")
print("=" * 60)

# Prepare data for visualization (only valid subcategories)
categories = []
counts = []
percentages = []
instance_percentages = []

for col in category_columns:
    if col in category_counts:
        categories.append(display_names.get(col, col))
        counts.append(category_counts[col])
        percentages.append((category_counts[col] / len(df_clean)) * 100)
        instance_percentages.append(
            (category_counts[col] / total_category_instances) * 100
        )

# Sort by count (descending)
sorted_indices = np.argsort(counts)[::-1]
categories = [categories[i] for i in sorted_indices]
counts = [counts[i] for i in sorted_indices]
percentages = [percentages[i] for i in sorted_indices]
instance_percentages = [instance_percentages[i] for i in sorted_indices]

# Set up the plot with custom font
plt.rcParams["font.family"] = "Century Gothic"
plt.rcParams["font.size"] = 12

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar chart
y_positions = np.arange(len(categories))
bars = ax.barh(y_positions, counts, height=0.6)

# Apply colors, using gradient from primary colors
colors_to_use = [
    COLORS["primary"][0],  # Dark blue
    COLORS["primary"][1],  # Teal
    COLORS["secondary"][0],  # Light blue
    COLORS["secondary"][1],
    COLORS["others"][5],
]

for bar, color in zip(bars, colors_to_use[: len(bars)]):
    bar.set_color(color)
    bar.set_edgecolor("white")
    bar.set_linewidth(2)

# Add value labels on the bars
threshold = 0.15 * max(counts)  # 15 % of the longest bar

for i, (count, pct, inst_pct) in enumerate(
    zip(counts, percentages, instance_percentages)
):
    if count < threshold:
        # ---- SHORT bar ----
        # count (closer to the bar)
        ax.text(
            count + max(counts) * 0.02,
            i,
            f"{count:,}",
            ha="left",
            va="center",
            color="#182A48",
            fontweight="bold",
            fontsize=12,
        )
        # percentage (use instance percentage)
        ax.text(
            count + max(counts) * 0.10,
            i,
            f"({inst_pct:.1f}%)",
            ha="left",
            va="center",
            color="#182A48",
            fontsize=11,
        )
    else:
        # ---- LONG bar ----
        ax.text(
            count / 2,
            i,
            f"{count:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=14,
        )
        ax.text(
            count + max(counts) * 0.01,
            i,
            f"({inst_pct:.1f}%)",
            ha="left",
            va="center",
            color="#182A48",
            fontsize=11,
        )

# Customize the plot
ax.set_yticks(y_positions)
ax.set_yticklabels(categories, fontsize=13)
ax.set_xlabel("Number of Documents", fontsize=14, fontweight="bold")
ax.set_title(
    "Distribution of Heat Pump Innovation Categories",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Add grid for better readability
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Set x-axis limits with some padding
ax.set_xlim(0, max(counts) * 1.15)
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

# CHANGE: Updated subtitle to include category instances with careful spacing
plt.text(
    0.5,
    -0.15,
    f"Total documents analysed: {len(df_clean):,} | "
    f"Total documents plotted: {len(documents_with_valid_categories):,} | "
    f"Total category instances: {total_category_instances:,}\n"
    f"Documents may belong to multiple categories",
    transform=ax.transAxes,
    ha="center",
    fontsize=11,
    style="italic",
)

# Adjust layout with more bottom margin to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 1])

# Save the plots
output_dir = Path("/home/pascualdiego/projects/DiscoveryHP/outputs/Plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Save as PNG
png_path = output_dir / "02_innovation_category_distribution.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved PNG: {png_path}")

# Save as PDF
pdf_path = output_dir / "02_innovation_category_distribution.pdf"
plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
print(f"Saved PDF: {pdf_path}")

# Show the plot
plt.show()

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

# Create a summary statistics table
print("\nSummary Statistics:")
summary_df = pd.DataFrame(
    {
        "Category": [cat.replace("\n", " ") for cat in categories],
        "Count": [f"{count:,}" for count in counts],
        "Percentage of Instances": [
            f"{pct:.1f}%" for pct in instance_percentages
        ],
    }
)
print(summary_df.to_string(index=False))

# Create summary text file
print("\n" + "=" * 60)
print("CREATING SUMMARY TEXT FILE")
print("=" * 60)

summary_text = f"""Heat Pump Innovation Category Distribution Analysis
===================================================
Dataset Overview:
-----------------
Total number of documents analyzed: {len(df_clean):,}
Total number of documents plotted (with valid subcategories): {len(documents_with_valid_categories):,}
Total number of documents with no categories: {len(documents_with_no_categories):,}
Total category instances: {total_category_instances:,}

Main Reason for Variation (Analyzed vs Plotted):
------------------------------------------------
The difference between analyzed ({len(df_clean):,}) and plotted ({len(documents_with_valid_categories):,}) documents is due to:
1. Documents with no innovation categories: {len(documents_with_no_categories):,} documents - These documents have empty values ('[]' or missing) for all category fields
2. Documents with invalid subcategories (excluded per category):
"""

# Add excluded counts per category
total_excluded_subcategory = sum(excluded_counts.values())
for col in category_columns:
    if excluded_counts[col] > 0:
        display_name = display_names.get(col, col).replace("\n", " ")
        summary_text += (
            f" - {display_name}: {excluded_counts[col]:,} documents excluded\n"
        )

summary_text += f"""
Total documents excluded due to invalid subcategories: {total_excluded_subcategory:,}
(Note: A document may be excluded from multiple categories)

Category Instance Explanation:
------------------------------
The percentages shown in the visualization are calculated based on the total number of
category instances ({total_category_instances:,}), not the total number of documents.
Since documents can belong to multiple categories, the sum of all percentages equals 100% of category instances, not 100% of documents.
This approach provides a clearer view of how innovation efforts are distributed across different categories.

Innovation Category Distribution (Valid Subcategories Only):
------------------------------------------------------------
"""

# Add category statistics
for i, (cat, count, inst_pct) in enumerate(
    zip(categories, counts, instance_percentages)
):
    cat_display = cat.replace("\n", " ")
    summary_text += (
        f"{cat_display}: {count:,} documents ({inst_pct:.1f}% of category instances)\n"
    )

# Add comparison with total counts
summary_text += """
Comparison with Total Category Counts (before taxonomy filtering):
------------------------------------------------------------------
"""
for col in category_columns:
    display_name = display_names.get(col, col).replace("\n", " ")
    total = category_counts_total[col]
    valid = category_counts[col]
    excluded = excluded_counts[col]
    summary_text += (
        f"{display_name}: {total:,} total \u2192 {valid:,} valid ({excluded:,} excluded)\n"
    )

# Add information about other categories if found
if other_category_counts:
    summary_text += """
Other Categories Found in Dataset:
----------------------------------
"""
    for cat, count in other_category_counts.items():
        summary_text += f"{cat}: {count:,} documents\n"

summary_text += """
Multi-Category Analysis:
------------------------
"""
for num_cats, count in multi_category_counts.items():
    summary_text += f"Documents with {num_cats} valid categories: {count:,}\n"

# Calculate average categories per document
avg_categories = (
    total_category_instances / len(documents_with_valid_categories)
    if len(documents_with_valid_categories) > 0
    else 0
)
summary_text += f"\nAverage categories per document: {avg_categories:.2f}\n"

summary_text += f"""
Visualization Description for Thesis:
-------------------------------------
This horizontal bar chart reveals the distribution of heat pump innovations across five main categories, filtered to include only documents with subcategories matching the established taxonomy. The percentages shown represent the proportion of total category instances, providing a normalized view of innovation focus areas.

Key Insights for the Thesis:
1. Innovation Distribution: With {total_category_instances:,} category instances across {len(documents_with_valid_categories):,} documents, the average document addresses {avg_categories:.2f} innovation categories, indicating that heat pump innovations often take integrated approaches combining multiple improvement areas.
2. Traditional Components Dominance: Traditional Components represent the largest share of innovation instances, confirming that the industry continues to prioritize incremental improvements to established technologies. This focus on proven components reflects both market maturity and the critical importance of reliability in heating systems.
3. Circular Economy Gap: The Circular Economy category shows the lowest representation among innovation instances. This gap is particularly significant given that each heat pump unit contains valuable materials and components that could benefit from circular design principles. The low percentage suggests substantial untapped potential.
4. System-Level Thinking: System Design innovations represent a meaningful portion of instances, indicating growing recognition that heat pump performance depends not just on individual components but on holistic system integration, a principle aligned with circular economy thinking.
5. Multi-Category Innovation: The fact that documents average {avg_categories:.2f} categories suggests that many innovations are taking integrated approaches. This is encouraging for circular economy adoption, as circular principles often require changes across multiple aspects of a product.
6. Research Validation: The {len(documents_with_valid_categories):,} documents with valid subcategories ({(len(documents_with_valid_categories)/len(df_clean)*100):.1f}% of total) provide a robust dataset for analyzing established innovation patterns, while the {total_excluded_subcategory:,} excluded category instances point to emerging trends outside current taxonomies.

This filtered analysis provides confidence in the innovation patterns identified while highlighting areas where new approaches, particularly those aligned with circular economy principles, may be emerging outside traditional classifications. The instance-based percentages offer a clearer picture of where innovation efforts are truly concentrated.

ANNEX: Excluded Document IDs by Category
========================================
Documents with Invalid Subcategories:
-------------------------------------
"""

# Add excluded documents per category
for col in category_columns:
    if excluded_documents_by_category[col]:
        display_name = display_names.get(col, col).replace("\n", " ")
        summary_text += (
            f"\n{display_name} ({len(excluded_documents_by_category[col]):,} documents):\n"
        )
        summary_text += "-" * 50 + "\n"
        # Group IDs in lines of 10 for readability
        for i in range(0, len(excluded_documents_by_category[col]), 10):
            batch = excluded_documents_by_category[col][i : i + 10]
            summary_text += ", ".join(str(doc_id) for doc_id in batch) + "\n"

summary_text += f"""
Documents with No Categories:
-----------------------------
Total: {len(documents_with_no_categories):,} documents
Document IDs:
"""

# Add no-category document IDs
for i in range(0, len(documents_with_no_categories), 10):
    batch = documents_with_no_categories[i : i + 10]
    summary_text += ", ".join(str(doc_id) for doc_id in batch) + "\n"

# Save summary file
summary_path = output_dir / "02_innovation_category_distribution_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"Summary saved to: {summary_path}")

print("\nSummary preview:")
print(summary_text[:500] + "...")

print("\n" + "=" * 60)
print("SCRIPT EXECUTION COMPLETE")
print("=" * 60)
