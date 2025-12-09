#!/usr/bin/env python3
"""
Heat-Pump Innovation Analysis – Plot 1B:
Heat-Pump-Type Distribution Pie Chart
Updated with additional analysis for ASHP and domestic applications
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# 0. SETTINGS
# ────────────────────────────────────────────────────────────────
DATA_PATHS = [
    "/home/pascualdiego/projects/DiscoveryHP/inputs/heat_pump_data_w_categories.csv",
    "inputs/heat_pump_data_w_categories.csv",
]
HP_TYPE_COL = "heat_pump_type"          # ← change here if your column is named differently
FALLBACK_COL = "heat_pump_type"         # automatic fallback alias
APP_TYPE_COL = "application_type"       # for domestic analysis

# Distinct colour palette (colour-blind friendly)
HP_COLOURS = [
    "#4E79A7", "#76B7B2", "#F28E2B", "#E15759",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"
]

plt.rcParams['font.family'] = ['Century Gothic', 'Arial']
plt.rcParams['font.size']   = 12


# ────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────────────────────
for p in DATA_PATHS:
    try:
        df = pd.read_csv(p)
        print(f"✓  Loaded data from: {p}")
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError("❌  Could not locate input CSV in the provided paths.")

# ADDITION: Track original document count
documents_analyzed = len(df)

# Detect column name automatically if needed
if HP_TYPE_COL not in df.columns and FALLBACK_COL in df.columns:
    HP_TYPE_COL = FALLBACK_COL
    print(f"ℹ️  Using fallback column name: “{HP_TYPE_COL}”")

if HP_TYPE_COL not in df.columns:
    raise KeyError(f"Column “{HP_TYPE_COL}” not found in dataset!")


# ────────────────────────────────────────────────────────────────
# 2. QUICK CLEAN
# ────────────────────────────────────────────────────────────────
# Keep all columns for cross-analysis
df_clean = df.copy()

# Clean heat pump type column
df_clean[HP_TYPE_COL] = (df_clean[HP_TYPE_COL]
                        .fillna('unclear')
                        .str.lower()
                        .str.strip())

# Clean application type if exists
if APP_TYPE_COL in df_clean.columns:
    df_clean[APP_TYPE_COL] = (df_clean[APP_TYPE_COL]
                             .fillna('unclear')
                             .str.lower()
                             .str.strip())

# Keep only the four heat-pump types we care about
EXPECTED_HP_TYPES = {'unclear', 'ashp', 'gshp', 'both'}

unexpected_hp = sorted(set(df_clean[HP_TYPE_COL]) - EXPECTED_HP_TYPES)
if unexpected_hp:
    print(f"\n⚠️  Excluding unexpected heat-pump types from plots: {unexpected_hp}")

df_clean = df_clean[df_clean[HP_TYPE_COL].isin(EXPECTED_HP_TYPES)].copy()

print("\nFiltered HP-type value counts (used in pie):")
print(df_clean[HP_TYPE_COL].value_counts())


# ────────────────────────────────────────────────────────────────
# 3. PIE CHART
# ────────────────────────────────────────────────────────────────
type_counts = df_clean[HP_TYPE_COL].value_counts()

# CHANGE 2: Define specific order and full names
hp_order = ['ashp', 'gshp', 'both', 'unclear']
hp_display_names = {
    'ashp': 'Air Source Heat Pump (ASHP)',
    'gshp': 'Ground Source Heat Pump (GSHP)',
    'both': 'Both',
    'unclear': 'Unclear'
}

hp_display_names_2 = {
    'ashp': 'Air Source\nHeat Pump (ASHP)',
    'gshp': 'Ground Source\nHeat Pump (GSHP)',
    'both': 'Both',
    'unclear': 'Unclear'
}

# Reorder data according to specific order
ordered_labels = []
ordered_sizes = []
ordered_colors = []

for i, hp_type in enumerate(hp_order):
    if hp_type in type_counts.index:
        ordered_labels.append(hp_display_names[hp_type])
        ordered_sizes.append(type_counts[hp_type])
        ordered_colors.append(HP_COLOURS[i])

fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    ordered_sizes,
    colors     = ordered_colors,
    labels     = None,          # legend below instead
    autopct    = '%1.1f%%',
    startangle = 90,
    explode    = [0.04] * len(ordered_labels)
)

# CHANGE 2: Create 2x2 legend layout
legend_elements = []
for label, color in zip(ordered_labels, ordered_colors):
    legend_elements.append(mpatches.Rectangle((0, 0), 1, 1, fc=color, label=label))

# Create legend with 2 columns to achieve the desired layout
ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    frameon=False,
    fontsize=13,
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=3
)

# Text styling
for at in autotexts:
    at.set_color('white')
    at.set_weight('bold')
    at.set_fontsize(13)

ax.set_title('Distribution of Documents by Heat Pump Type',
             fontsize=16, weight='bold', pad=20)
ax.axis('equal')

# CHANGE 1: Updated footer with both analyzed and plotted
fig.text(0.5, -0.2,
         f'Total documents analysed: {documents_analyzed:,} | Total documents plotted: {len(df_clean):,}',
         transform=ax.transAxes,
         ha='center', va='bottom',
         fontsize=13, style='italic')

plt.tight_layout()

# Save pie chart
output_dir = Path("/home/pascualdiego/projects/DiscoveryHP/outputs/Plots")
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "01b_hp_type_distribution_pie.png"
pdf_path = output_dir / "01b_hp_type_distribution_pie.pdf"

fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')

print(f"\n✓  Saved PNG: {png_path}")
print(f"✓  Saved PDF: {pdf_path}")

plt.show()


# ────────────────────────────────────────────────────────────────
# CHANGE 3: ADDITIONAL PLOT - ASHP & DOMESTIC ANALYSIS
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CREATING ASHP & DOMESTIC APPLICATION ANALYSIS")
print("="*60)

if APP_TYPE_COL in df_clean.columns:
    # Create cross-tabulation
    crosstab = pd.crosstab(df_clean[HP_TYPE_COL], df_clean[APP_TYPE_COL])
    
    # Calculate percentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    # Create stacked bar chart
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Define colors for application types
    app_colors = {
        'domestic':  '#182A48',  # dark navy
        'industrial': '#18A48C', # teal
        'both':       '#A59BEE',  # lavender
        'unclear':   '#D2C9C0',  # warm grey
    }
    
    # Reorder heat pump types for display
    hp_types_ordered = []
    for hp in hp_order:
        if hp in crosstab.index:
            hp_types_ordered.append(hp)
    
    # Create the stacked bar chart
    bottom = None
    for app_type in ['domestic', 'industrial', 'both', 'unclear']:
        if app_type in crosstab.columns:
            values = []
            for hp in hp_types_ordered:
                values.append(crosstab_pct.loc[hp, app_type] if hp in crosstab_pct.index else 0)
            
            bars = ax2.bar(range(len(hp_types_ordered)), values, 
                          bottom=bottom,
                          label=app_type.capitalize(),
                          color=app_colors.get(app_type, '#999999'))
            
            # Add percentage labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 5:  # Only show label if >5%
                    height = bar.get_height()
                    y_pos = bar.get_y() + height/2
                    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{val:.0f}%',
                           ha='center', va='center',
                           color='white', fontweight='bold', fontsize=10)
            
            bottom = values if bottom is None else [b+v for b,v in zip(bottom, values)]
    
    # Customize plot
    ax2.set_xticks(range(len(hp_types_ordered)))
    ax2.set_xticklabels([hp_display_names_2[hp] for hp in hp_types_ordered], 
                       rotation=0, ha='center')
    ax2.set_ylabel('Percentage of Documents (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Heat Pump Types by Application Context',
                 fontsize=16, fontweight='bold', pad=20)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
# Add statistics text (sample size for each bar)
    ashp_domestic = crosstab.loc['ashp', 'domestic'] if 'ashp' in crosstab.index and 'domestic' in crosstab.columns else 0
    ashp_total    = crosstab.loc['ashp'].sum()    if 'ashp'    in crosstab.index else 0
    gshp_total    = crosstab.loc['gshp'].sum()    if 'gshp'    in crosstab.index else 0
    both_total    = crosstab.loc['both'].sum()    if 'both'    in crosstab.index else 0
    unclear_total = crosstab.loc['unclear'].sum() if 'unclear' in crosstab.index else 0

    ashp_domestic_pct = (ashp_domestic / ashp_total * 100) if ashp_total > 0 else 0

    stats_text = (
        f'ASHP documents: {ashp_total:,} | '
        f'GSHP documents: {gshp_total:,} |\n ' 
        f'Both documents: {both_total:,} | '
        f'Unclear documents: {unclear_total:,} .\n '
        f'ASHP in domestic applications: {ashp_domestic:,} ({ashp_domestic_pct:.1f}%)'
    )

    ax2.text(0.5, -0.22, stats_text,
             transform=ax2.transAxes, ha='center', fontsize=13, style='italic')

    # First let tight_layout arrange things, then give extra space at the bottom
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.25)  # increase if the text is still too close
    
    # Save additional plot
    png_path2 = output_dir / "01b_hp_type_by_application.png"
    pdf_path2 = output_dir / "01b_hp_type_by_application.pdf"
    
    fig2.savefig(png_path2, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(pdf_path2, format='pdf', bbox_inches='tight', facecolor='white')
    
    print(f"✓  Saved PNG: {png_path2}")
    print(f"✓  Saved PDF: {pdf_path2}")
    
    plt.show()
else:
    print("⚠️  Application type column not found - skipping ASHP/domestic analysis plot")


# ────────────────────────────────────────────────────────────────
# CHANGE 4: CREATE SUMMARY TEXT FILE
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CREATING SUMMARY TEXT FILE")
print("="*60)

# Calculate statistics
ashp_count = type_counts.get('ashp', 0)
ashp_percentage = (ashp_count / len(df_clean) * 100) if len(df_clean) > 0 else 0

# Get domestic statistics if available
if APP_TYPE_COL in df_clean.columns and 'ashp' in crosstab.index and 'domestic' in crosstab.columns:
    ashp_domestic = crosstab.loc['ashp', 'domestic']
    total_domestic = crosstab['domestic'].sum()
    ashp_of_domestic = (ashp_domestic / total_domestic * 100) if total_domestic > 0 else 0
else:
    ashp_domestic = 0
    total_domestic = 0
    ashp_of_domestic = 0

summary_text = f"""Heat Pump Type Distribution Analysis
=====================================

Dataset Overview:
-----------------
Total number of documents analyzed: {documents_analyzed}
Total number of documents plotted: {len(df_clean)}
Documents excluded: {documents_analyzed - len(df_clean)}

Main Reason for Variation (Analyzed vs Plotted):
------------------------------------------------
The difference of {documents_analyzed - len(df_clean)} documents is due to:
1. Documents with unexpected heat pump type values: {len(unexpected_hp) if unexpected_hp else 0}
   - Only the four standard types (ASHP, GSHP, Both, Unclear) are included
   - Excluded values: {', '.join(unexpected_hp) if unexpected_hp else 'None'}
2. This ensures consistency in analyzing recognized heat pump technologies

Heat Pump Type Distribution:
----------------------------
"""

# Add statistics for each type
for hp_type in hp_order:
    if hp_type in type_counts.index:
        count = type_counts[hp_type]
        percentage = (count / len(df_clean) * 100)
        summary_text += f"{hp_display_names[hp_type]}: {count} documents ({percentage:.1f}%)\n"

summary_text += f"""
Air Source Heat Pump (ASHP) Focus:
----------------------------------
Total ASHP documents: {ashp_count} ({ashp_percentage:.1f}% of all documents)
"""

if APP_TYPE_COL in df_clean.columns:
    summary_text += f"""ASHP in domestic applications: {ashp_domestic} documents
ASHP represents {ashp_of_domestic:.1f}% of all domestic heat pump documents
Domestic applications represent {ashp_domestic_pct:.1f}% of all ASHP documents
"""

summary_text += f"""
Visualization Description for Thesis:
-------------------------------------
This heat pump type distribution analysis provides critical insights into the technological 
landscape of heat pump innovations, with particular relevance to Air Source Heat Pumps (ASHP) 
and their dominance in domestic applications.

Key Insights for ASHP and Domestic Applications:

1. ASHP Market Dominance: Air Source Heat Pumps represent {ashp_percentage:.1f}% of the 
   innovation landscape, confirming their position as the most researched and developed 
   heat pump technology. This dominance reflects several factors:
   - Lower installation costs compared to GSHP
   - Easier retrofit potential in existing buildings
   - Broader applicability across different property types
   - Reduced site requirements (no ground drilling needed)

2. ASHP-Domestic Connection: The strong correlation between ASHP and domestic applications 
   is particularly significant for residential decarbonization strategies. ASHPs are the 
   primary technology for domestic heating transitions because:
   - They require minimal property modifications
   - Installation is less disruptive than ground source alternatives
   - They're suitable for urban and suburban environments where most homes are located
   - They offer a more affordable entry point for homeowners

3. Innovation Implications: The concentration of innovation in ASHP technology suggests:
   - Market maturity in addressing technical challenges (efficiency, cold climate operation)
   - Industry recognition of ASHP as the scalable solution for mass deployment
   - Continued need for innovations in noise reduction, aesthetics, and integration
   - Opportunities for circular economy principles in the dominant technology

4. Ground Source Heat Pump (GSHP) Niche: While GSHPs show lower representation, they 
   remain important for specific applications (large buildings, new developments) where 
   their higher efficiency can be fully utilized.

5. Technology Uncertainty: The "unclear" category ({type_counts.get('unclear', 0)} documents) 
   may represent:
   - System-level innovations applicable to multiple heat pump types
   - Emerging hybrid technologies
   - Research focusing on components rather than complete systems

6. Circular Economy Opportunities: With ASHP dominating the domestic market, implementing 
   circular economy principles in ASHP design and manufacturing could have the greatest 
   impact on residential heating sustainability. This includes:
   - Designing ASHPs for easy maintenance and repair
   - Developing take-back schemes for the millions of units expected to be installed
   - Creating modular designs allowing component upgrades
   - Establishing recycling pathways for ASHP-specific components

The overwhelming focus on ASHP in the innovation landscape, particularly for domestic 
applications, validates strategies targeting this technology for achieving residential 
heating decarbonization at scale. However, it also highlights the critical need to 
ensure these dominant technologies are designed with circular economy principles from 
the outset to avoid creating future waste streams as millions of units are deployed.
"""

# Save summary file
summary_path = output_dir / "01b_heat_pump_type_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"Summary saved to: {summary_path}")
print("\nSummary preview:")
print(summary_text[:500] + "...")


# ────────────────────────────────────────────────────────────────
# 5. FINAL SUMMARY PRINT
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nHeat Pump Type Distribution:")
for hp_type in hp_order:
    if hp_type in type_counts.index:
        count = type_counts[hp_type]
        pct = count/len(df_clean)*100
        print(f"  {hp_display_names[hp_type]}: {count} documents ({pct:.1f}%)")

if APP_TYPE_COL in df_clean.columns:
    print(f"\nASHP Domestic Focus:")
    print(f"  ASHP in domestic applications: {ashp_domestic} documents")
    print(f"  This represents {ashp_domestic_pct:.1f}% of all ASHP documents")

print("\n" + "="*60)
print("SCRIPT EXECUTION COMPLETE")
print("="*60)