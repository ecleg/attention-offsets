#!/usr/bin/env python3
"""
Sankey diagram for ChatGPT Environmental Impact Analysis.
"""

import plotly.graph_objects as go

# =============================================================================
# COLOR PALETTE FROM colors.jpg
# =============================================================================
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CHANGE COLORS HERE - Edit hex values below to customize the diagram       ║
# ╚════════════════════════════════════════════════════════════════════════════╝
COLORS = {
    "input": "#565fa2",        # Purple/Indigo - Starting node
    "extraction": "#0A616F",   # Teal/Cyan - Data extraction nodes
    "models": "#fa7317",       # Deep Orange - AI model nodes (gpt-4o, o1, etc.)
    "tools": "#00bcd4",        # CYAN - Tool nodes (dalle, python, browser, etc.)
    "derived": "#efc227",      # Gold - Derived values (Message Counts, Tokens, etc.)
    "calculations": "#53a155", # Green - Environmental calculations (CO2, Energy)
    "water": "#3a28bf",        # Teal - Water-related nodes
    "output": "#2a2e4e",       # Purple - Final output node
    "highlight": "#A79331",    # Light Gold - Key metrics (Offset Cost, Ad Revenue)
}
# To change a color: Replace the hex code (e.g., "#00bcd4") with your desired color
# Hex color picker: https://htmlcolorcodes.com/

# =============================================================================
# REAL DATA FROM USER ANALYSIS (6 users, 5,055 prompts)
# =============================================================================
REAL_DATA = {
    "avg_conversations": 91.8,
    "avg_messages": 2141.8,
    "avg_user_prompts": 842.5,
    "avg_tokens": 328692,
    "avg_weighted_kwh": 4.6007,
    "avg_co2_kg": 1.6102,
    "avg_water_liters": 8.28,
    "avg_ad_revenue": 4.2125,           # $4.21 per user
    "avg_carbon_offset": 0.016102,      # $0.016 per user
    "avg_water_offset": 0.003281,       # $0.003 per user
    "avg_combined_offset": 0.019384,    # $0.019 per user
    "coverage_ratio": 217.3,            # 4.2125 / 0.019384
    "water_pct_of_offset": 16.9,
    "carbon_pct_of_offset": 83.1,
}

# =============================================================================
# NODE DEFINITIONS
# =============================================================================

nodes = [
    # === COLUMN 0: INPUT (1 node) ===
    "ChatGPT Export Data",                    # 0
    
    # === COLUMN 1: EXTRACTION (4 nodes) ===
    "Conversation ID",                        # 1
    "Title",                                  # 2
    "Timestamp",                              # 3
    "Messages Array",                         # 4
    
    # === COLUMN 2: MESSAGE FIELDS (4 nodes) ===
    "Content Text",                           # 5
    "Author Role",                            # 6
    "Model Identifier",                       # 7
    "Tool Recipient",                         # 8
    
    # === COLUMN 3: TOOLS FIRST (8 nodes) - Cyan, positioned ABOVE models ===
    "dalle.text2im (15.0×)",                  # 9  (was 17)
    "python (1.2×)",                          # 10 (was 18)
    "browser (1.1×)",                         # 11 (was 19)
    "web.run (1.1×)",                         # 12 (was 20)
    "file_search (1.3×)",                     # 13 (was 21)
    "canmore.canvas (1.0×)",                  # 14 (was 22)
    "code_interpreter (1.2×)",                # 15 (was 23)
    "No Tool (1.0×)",                         # 16 (was 24)
    
    # === COLUMN 3: MODELS SECOND (8 nodes) - Deep Orange, positioned BELOW tools ===
    "gpt-3.5-turbo (1.0×)",                   # 17 (was 9)
    "gpt-4 (2.0×)",                           # 18 (was 10)
    "gpt-4-turbo (2.0×)",                     # 19 (was 11)
    "gpt-4o (1.8×)",                          # 20 (was 12)
    "gpt-4o-mini (1.2×)",                     # 21 (was 13)
    "o1 (3.0×)",                              # 22 (was 14)
    "o1-mini (2.0×)",                         # 23 (was 15)
    "o1-preview (3.0×)",                      # 24 (was 16)
    
    # === COLUMN 4: DERIVED (4 nodes) ===
    "Message Counts",                         # 25
    "Estimated Tokens",                       # 26
    "Model Usage Distribution",               # 27
    "Tool Usage Distribution",                # 28
    
    # === COLUMN 5: ENERGY (2 nodes) ===
    "Weighted Energy Multiplier",             # 29
    "Avg Energy (4.6 kWh/user)",              # 30
    
    # === COLUMN 6: ENVIRONMENTAL IMPACT (split) ===
    "Avg CO₂ (1.6 kg/user)",                  # 31
    "Avg Water (8.3 L/user)",                 # 32
    
    # === COLUMN 7: COMPARISONS & OFFSETS ===
    "Real-World Comparisons",                 # 33
    "Avg Carbon Offset ($0.016)",             # 34
    "Avg Water Offset ($0.003)",              # 35
    
    # === COLUMN 8: COMBINED & REVENUE ===
    "Avg Combined Offset ($0.019)",           # 36
    "Avg Ad Revenue ($4.21)",                 # 37
    
    # === COLUMN 9: OUTPUT ===
    "Analysis",                               # 38
]

# =============================================================================
# NODE COLORS
# =============================================================================

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def lighten(hex_color, factor):
    r, g, b = hex_to_rgb(hex_color)
    return '#{:02x}{:02x}{:02x}'.format(
        int(min(255, r + (255 - r) * factor)),
        int(min(255, g + (255 - g) * factor)),
        int(min(255, b + (255 - b) * factor)))

def darken(hex_color, factor):
    r, g, b = hex_to_rgb(hex_color)
    return '#{:02x}{:02x}{:02x}'.format(
        int(r * (1-factor)), int(g * (1-factor)), int(b * (1-factor)))

# Category assignments (TOOLS first at 9-16, MODELS second at 17-24)
node_categories = [
    "input",       # 0
    "extraction", "extraction", "extraction", "extraction",  # 1-4
    "extraction", "extraction", "extraction", "extraction",  # 5-8
    "tools", "tools", "tools", "tools", "tools", "tools", "tools", "tools",  # 9-16 (TOOLS - Cyan)
    "models", "models", "models", "models", "models", "models", "models", "models",  # 17-24 (MODELS - Orange)
    "derived", "derived", "derived", "derived",  # 25-28
    "calculations", "calculations",  # 29-30
    "calculations", "water",  # 31-32 (CO2 green, Water teal)
    "calculations", "calculations", "water",  # 33-35
    "highlight", "highlight",  # 36-37 (combined offset, ad revenue)
    "output",  # 38
]

# Generate colors with gradients
node_colors = []
STEP = 0.06
category_positions = {}
for i, cat in enumerate(node_categories):
    if cat not in category_positions:
        category_positions[cat] = []
    category_positions[cat].append(i)

for i, cat in enumerate(node_categories):
    base = COLORS[cat]
    positions = category_positions[cat]
    idx = positions.index(i)
    total = len(positions)
    
    if total == 1:
        node_colors.append(base)
    else:
        mid = (total - 1) / 2
        offset = (idx - mid) * STEP
        if offset < 0:
            node_colors.append(lighten(base, abs(offset)))
        else:
            node_colors.append(darken(base, offset))

# =============================================================================
# LINK DEFINITIONS
# =============================================================================

links = {"source": [], "target": [], "value": [], "color": []}

def add_link(src, tgt, val, opacity=0.4):
    links["source"].append(src)
    links["target"].append(tgt)
    links["value"].append(val)
    r, g, b = hex_to_rgb(node_colors[src])
    links["color"].append(f"rgba({r}, {g}, {b}, {opacity})")

# === INPUT → EXTRACTION ===
add_link(0, 1, 6)    # → Conversation ID
add_link(0, 2, 6)    # → Title
add_link(0, 3, 6)    # → Timestamp
add_link(0, 4, 30)   # → Messages Array (main flow)

# === EXTRACTION → MESSAGE FIELDS ===
add_link(4, 5, 20)   # Messages → Content Text
add_link(4, 6, 10)   # Messages → Author Role
add_link(4, 7, 10)   # Messages → Model Identifier
add_link(4, 8, 8)    # Messages → Tool Recipient

# === TOOL RECIPIENT → TOOLS (indices 9-16, CYAN, positioned ABOVE models) ===
add_link(8, 9, 3)    # → dalle.text2im
add_link(8, 10, 5)   # → python
add_link(8, 11, 2)   # → browser
add_link(8, 12, 2)   # → web.run
add_link(8, 13, 1.5) # → file_search
add_link(8, 14, 1.5) # → canmore.canvas
add_link(8, 15, 3)   # → code_interpreter
add_link(8, 16, 15)  # → No Tool (majority)

# === MODEL IDENTIFIER → MODELS (indices 17-24, ORANGE, positioned BELOW tools) ===
add_link(7, 17, 2)   # → gpt-3.5-turbo
add_link(7, 18, 3)   # → gpt-4
add_link(7, 19, 2)   # → gpt-4-turbo
add_link(7, 20, 10)  # → gpt-4o (most common)
add_link(7, 21, 8)   # → gpt-4o-mini
add_link(7, 22, 2)   # → o1
add_link(7, 23, 1.5) # → o1-mini
add_link(7, 24, 1.5) # → o1-preview

# === CONTENT TEXT → MESSAGE COUNTS → ESTIMATED TOKENS ===
# Fixed: Message Counts flows INTO Estimated Tokens
add_link(6, 25, 15)   # Author Role → Message Counts
add_link(5, 25, 10)   # Content Text → Message Counts (counting messages)
add_link(25, 26, 20)  # Message Counts → Estimated Tokens (tokens derived from message content)
add_link(5, 26, 8)    # Content Text → Estimated Tokens (direct char counting)

# === TOOLS → TOOL USAGE DISTRIBUTION (indices 9-16) ===
for i in range(9, 17):
    add_link(i, 28, 2.5)

# === MODELS → MODEL USAGE DISTRIBUTION (indices 17-24) ===
for i in range(17, 25):
    add_link(i, 27, 3)

# === DERIVED → WEIGHTED MULTIPLIER ===
add_link(27, 29, 20)  # Model Usage → Weighted Multiplier
add_link(28, 29, 12)  # Tool Usage → Weighted Multiplier

# === DERIVED → ENERGY ===
add_link(26, 30, 18)  # Estimated Tokens → Avg Energy
add_link(29, 30, 18)  # Weighted Multiplier → Avg Energy

# === ENERGY → CO2 AND WATER (split) ===
add_link(30, 31, 25)  # Avg Energy → Avg CO₂
add_link(30, 32, 10)  # Avg Energy → Avg Water

# === CO2 → COMPARISONS & CARBON OFFSET ===
add_link(31, 33, 15)  # CO₂ → Real-World Comparisons
add_link(31, 34, 20)  # CO₂ → Carbon Offset (83% of combined)

# === WATER → WATER OFFSET ===
add_link(32, 35, 10)  # Water → Water Offset (17% of combined)

# === OFFSETS → COMBINED OFFSET ===
# Proportional: Carbon is 83%, Water is 17%
add_link(34, 36, 20)  # Carbon Offset → Combined Offset
add_link(35, 36, 4)   # Water Offset → Combined Offset

# === COMBINED OFFSET → AD REVENUE (the key comparison) ===
# Ad revenue is 217x larger than combined offset
# Visual: Combined offset flows into Ad Revenue showing it's covered
add_link(36, 37, 24)  # Combined Offset → Ad Revenue

# === MESSAGE COUNTS → AD REVENUE (attention funding source) ===
# This shows the "attention" (user prompts) generating the revenue
add_link(25, 37, 60)  # Message Counts → Ad Revenue (the attention-funded portion)

# === ALL KEY METRICS → ANALYSIS ===
add_link(33, 38, 10)  # Real-World Comparisons → Analysis
add_link(36, 38, 8)   # Combined Offset → Analysis
add_link(37, 38, 25)  # Ad Revenue → Analysis (emphasized)

# Metadata to Analysis
add_link(1, 38, 4)    # Conversation ID → Analysis
add_link(2, 38, 4)    # Title → Analysis
add_link(3, 38, 4)    # Timestamp → Analysis

# =============================================================================
# CREATE FIGURE
# =============================================================================

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=18,
        thickness=22,
        line=dict(color="#2d2d2d", width=0.8),
        label=nodes,
        color=node_colors,
        hovertemplate='<b>%{label}</b><extra></extra>',
    ),
    link=dict(
        source=links["source"],
        target=links["target"],
        value=links["value"],
        color=links["color"],
        hovertemplate='%{source.label} → %{target.label}<extra></extra>',
    ),
)])

# Layout
fig.update_layout(
    title=dict(
        text="<b>ChatGPT Environmental Impact: Attention-Funded Offsets</b><br>" +
             "<sup>Analysis of 6 users (5,055 prompts) • Ad revenue covers carbon + water offsets by 217×</sup>",
        font=dict(family="Helvetica, Arial, sans-serif", size=22, color="#1f2937"),
        x=0.5,
        xanchor="center",
    ),
    font=dict(
        family="Helvetica, Arial, sans-serif",
        size=10,
        color="#374151",
    ),
    paper_bgcolor="#fafafa",
    plot_bgcolor="#fafafa",
    width=1800,
    height=1050,
    margin=dict(l=15, r=15, t=90, b=50),
    annotations=[
        # Key insight callout
        dict(
            x=0.88, y=0.12,
            xref="paper", yref="paper",
            text="<b>Key Finding:</b><br>" +
                 "Avg Ad Revenue: <b>$4.21</b>/user<br>" +
                 "Avg Carbon Offset: <b>$0.016</b>/user<br>" +
                 "Avg Water Offset: <b>$0.003</b>/user<br>" +
                 "Combined Offset: <b>$0.019</b>/user<br>" +
                 "Coverage: <b>217×</b>",
            showarrow=False,
            font=dict(size=10, color="#1f2937", family="Helvetica"),
            bgcolor="#eacf44",
            bordercolor="#2d2d2d",
            borderwidth=1,
            borderpad=8,
            opacity=0.95,
        ),
        # Legend
        dict(x=0.02, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Input/Output", showarrow=False, 
             font=dict(size=9, color=COLORS["input"], family="Helvetica")),
        dict(x=0.10, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Extraction", showarrow=False, 
             font=dict(size=9, color=COLORS["extraction"], family="Helvetica")),
        dict(x=0.18, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Models", showarrow=False, 
             font=dict(size=9, color=COLORS["models"], family="Helvetica")),
        dict(x=0.24, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Tools", showarrow=False, 
             font=dict(size=9, color=COLORS["tools"], family="Helvetica")),
        dict(x=0.30, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Derived", showarrow=False, 
             font=dict(size=9, color=COLORS["derived"], family="Helvetica")),
        dict(x=0.37, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Carbon Impact", showarrow=False, 
             font=dict(size=9, color=COLORS["calculations"], family="Helvetica")),
        dict(x=0.47, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Water Impact", showarrow=False, 
             font=dict(size=9, color=COLORS["water"], family="Helvetica")),
        dict(x=0.56, y=0.98, xref="paper", yref="paper", 
             text="<b>●</b> Key Metrics", showarrow=False, 
             font=dict(size=9, color=COLORS["highlight"], family="Helvetica")),
    ],
)

# =============================================================================
# SAVE
# =============================================================================

base_path = "/Users"

svg_path = f"{base_path}/data_flow_sankey.svg"
fig.write_image(svg_path, format="svg")
print(f"✓ SVG saved: {svg_path}")

png_path = f"{base_path}/data_flow_sankey.png"
fig.write_image(png_path, format="png", scale=2)
print(f"✓ PNG saved: {png_path}")

print("SANKEY DIAGRAM GENERATED")
