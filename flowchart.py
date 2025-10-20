from graphviz import Digraph

# Create flowchart
dot = Digraph("NBCSP_Flow", format="png")

# General style
dot.attr(rankdir="TB", splines="ortho", fontname="Arial")

# --- Node Styles ---
def add_node(name, label, color):
    dot.node(name, label,
             shape="box", style="rounded,filled",
             fontsize="12", fontname="Arial",
             fillcolor=color, color="black")

# --- Nodes ---
add_node("pop", "Eligible Population", "#7FB3D5")   # light blue
add_node("never", "Persistent Never-Screeners\nLetter Only Sent", "#F5B041") # orange
add_node("kit", "iFOBT Kit Recipients\nKit Sent", "#5DADE2")  # blue
add_node("proc", "Kits Returned & Processed\nAnalysis Performed", "#BB8FCE") # purple
add_node("pos", "Positive iFOBT Result", "#F39C12") # orange-yellow
add_node("col_no", "Colonoscopy - No Complications\n(98% of procedures)\n$1289", "#52BE80") # green
add_node("col_comp", "Colonoscopy - With Complications\n(2% of procedures)", "#EC7063") # red

# --- Edges ---
dot.edge("pop", "never", label="Letter only: $1.50", fontsize="10", fontcolor="dimgray")
dot.edge("pop", "kit", label="Letter + iFOBT kit: $10.00", fontsize="10", fontcolor="dimgray")
dot.edge("kit", "proc", label="Return & analysis\nBase: $13.50 | Incorrect (5%): $23.50", fontsize="10", fontcolor="dimgray")
dot.edge("proc", "pos", label="No additional cost", fontsize="10", style="dashed", color="darkorange")
dot.edge("pos", "col_no", label="→ GP + Colonoscopy", fontsize="10", fontcolor="dimgray")
dot.edge("pos", "col_comp", label="→ GP + Colonoscopy", fontsize="10", fontcolor="dimgray")

# Output
dot.render("NBCSP_flowchart", view=True)
