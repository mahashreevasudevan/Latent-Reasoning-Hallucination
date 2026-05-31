
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# CHART 1 — Latent Hallucination Profile

def chart1():
    data = [
        (8,   3, "Surface\nMismatch",       "unsupported_leap"),
        (22,  3, "Procedural\nTrap",        "unsupported_leap"),
        (40,  2, "Causal\nTrap",            "unsupported_leap"),
        (41,  2, "Procedural\nTrap",        "invented_step"),
        (70,  2, "Surface\nMismatch",       "wrong_method"),
        (108, 4, "Conditional\nDependency", "unsupported_leap"),
    ]
    vuln_list = ["Surface\nMismatch","Procedural\nTrap",
                 "Causal\nTrap","Conditional\nDependency"]
    vuln_colors = {
        "Surface\nMismatch":      "#d62728",
        "Procedural\nTrap":       "#ff7f0e",
        "Causal\nTrap":           "#bcbd22",
        "Conditional\nDependency":"#9467bd",
    }
    flaw_short = {
        "unsupported_leap":"leap",
        "invented_step":   "invented step",
        "wrong_method":    "wrong method",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(1.7, 4.3, color="lightyellow", alpha=0.8, zorder=0)
    ax.text(3.0, 3.75, "Hallucination zone (load 2-4)",
            ha='center', fontsize=8, color="goldenrod", style='italic')
    for i in range(len(vuln_list)):
        ax.axhline(i, color='lightgray', linewidth=0.8, zorder=1)
    for x in [1,2,3,4,5]:
        ax.axvline(x, color='lightgray', linewidth=0.8, zorder=1)

    placed = {}
    for (qid, load, vuln, flaw) in data:
        key = (load, vuln)
        placed[key] = placed.get(key, 0)
        offset = placed[key] * 0.15
        placed[key] += 1
        y = vuln_list.index(vuln)
        c = vuln_colors[vuln]
        ax.scatter(load, y+offset, s=260, color=c, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.text(load, y+offset, f"Q{qid}", ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=6)
        ax.text(load+0.18, y+offset+0.12, flaw_short[flaw],
                fontsize=7, color='gray', va='bottom')

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.6, 3.7)
    ax.set_xticks([1,2,3,4,5])
    ax.set_yticks(range(len(vuln_list)))
    ax.set_yticklabels(vuln_list, fontsize=9)
    ax.set_xlabel("Reasoning Load  (1 = trivial to 5 = maximal)", fontsize=10)
    ax.set_ylabel("Vulnerability Type", fontsize=10)
    ax.set_title("Latent Hallucination Profile", fontsize=12,
                 fontweight='bold', loc='left')
    handles = [mpatches.Patch(color=c, label=v.replace("\n"," "))
               for v,c in vuln_colors.items()]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart1_lh_profile.png", dpi=180, bbox_inches='tight')
    print("Saved: chart1_lh_profile.png")
    plt.close()


# CHART 2 — Reasoning Quality Gradient (AQuA-RAT)

def chart2():
    models = [
        ("Llama\n3.1 8B",  15, 15, 2,  13),
        ("GPT-OSS\n20B",   23,  7, 6,  17),
        ("Llama 4\nScout", 24,  6, 4,  20),
        ("Qwen3\n32B",     15, 15, 0,  15),
        ("Llama 3.3\n70B", 24,  6, 8,  16),
        ("GPT-OSS\n120B",  24,  6, 10, 14),
    ]
    labels  = [m[0] for m in models]
    flawed  = np.array([m[2] for m in models])
    shallow = np.array([m[3] for m in models])
    robust  = np.array([m[4] for m in models])
    correct = [m[1] for m in models]
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x, flawed,  width=0.55, label="Flawed",
                color="#d62728", alpha=0.85)
    b2 = ax.bar(x, shallow, width=0.55, bottom=flawed,
                label="Shallow-but-correct", color="#ff7f0e", alpha=0.85)
    b3 = ax.bar(x, robust,  width=0.55, bottom=flawed+shallow,
                label="Robust", color="#2ca02c", alpha=0.85)

    for i, c in enumerate(correct):
        ax.text(i, 31.2, f"{c}/30", ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    def label_bar(bars, bottoms, values):
        for bar, bot, val in zip(bars, bottoms, values):
            if val >= 3:
                ax.text(bar.get_x()+bar.get_width()/2, bot+val/2, str(val),
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')

    label_bar(b1, np.zeros(len(models)), flawed)
    label_bar(b2, flawed, shallow)
    label_bar(b3, flawed+shallow, robust)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0,5,10,15,20,25,30])
    ax.set_ylabel("AQuA-RAT responses  (n = 30)", fontsize=10)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title("Reasoning Quality Across Models (AQuA-RAT)",
                 fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart2_quality_gradient.png", dpi=180, bbox_inches='tight')
    print("Saved: chart2_quality_gradient.png")
    plt.close()


# CHART 3 — Dataset Taxonomy (Two-Condition Theory)

def chart3():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    col_labels = ["Dataset","Primary Stress",
                  "Answer\nRetrievable?","Non-trivial\nReasoning?",
                  "Failure Mode","Latent\nHallucinations"]
    rows = [
        ["GSM8K","Arithmetic retrieval",
         "x  No","x  No","Near-zero errors (all models >96%)","0"],
        ["AQuA-RAT","Algebraic manipulation",
         "x  No","v  Yes","Overt failures (wrong answer + flawed CoT)","0"],
        ["Custom Dataset","Cognitive traps",
         "v  Yes","v  Yes","Latent hallucinations (correct answer, wrong reasoning)","6"],
    ]
    col_widths = [0.11,0.17,0.11,0.11,0.38,0.11]

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   colWidths=col_widths, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.4)

    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor('#2c2c2c')
        tbl[0,j].set_text_props(color='white', fontweight='bold')

    row_bg = ["#eaf4fb","#fef5e7","#fdf0f0"]
    for i, row in enumerate(rows):
        for j in range(len(col_labels)):
            tbl[i+1,j].set_facecolor(row_bg[i])
            if j in [2,3]:
                tbl[i+1,j].set_text_props(
                    color="#2ca02c" if "v" in row[j] else "#d62728",
                    fontweight='bold')
            if j == 0:
                tbl[i+1,j].set_text_props(fontweight='bold')
            if i == 2 and j == 5:
                tbl[i+1,j].set_text_props(fontweight='bold', color='#d62728')

    ax.set_title("Dataset Taxonomy - Two-Condition Theory\n"
                 "Latent hallucinations require both conditions simultaneously",
                 fontsize=11, fontweight='bold', loc='left', pad=12)
    plt.tight_layout()
    plt.savefig("chart3_dataset_taxonomy.png", dpi=180, bbox_inches='tight')
    print("Saved: chart3_dataset_taxonomy.png")
    plt.close()


# CHART 4 — Cross-Dataset Accuracy Comparison

def chart4():
    model_labels = ["8B","20B","17B","32B","70B","120B"]
    x = np.arange(len(model_labels))

    series = [
        ("GSM8K",    [96.7,96.7,100, 100, 100, 100 ],[0,0,0,0,0,0],(0,(3,1,1,1))),
        ("AQuA-RAT", [50.0,76.7,80.0,50.0,80.0,80.0],[0,0,0,0,0,0],(0,(5,3))),
        ("Custom",   [62.7,82.2,93.3,98.0,100, 91.1],[6,0,0,0,0,0],'solid'),
    ]
    ds_colors = {
        "GSM8K":    "#1f77b4",
        "AQuA-RAT": "#ff7f0e",
        "Custom":   "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for ds_name, accs, lhs, ls in series:
        col = ds_colors[ds_name]
        ax.plot(x, accs, color=col, linewidth=2.2, linestyle=ls,
                zorder=3, label=ds_name)
        for i, (acc, lh) in enumerate(zip(accs, lhs)):
            if lh > 0:
                ax.scatter(x[i], acc, s=160, color=col, zorder=5,
                           edgecolors='white', linewidths=1.5)
                ax.text(x[i], acc, str(lh), ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold', zorder=6)
            else:
                ax.scatter(x[i], acc, s=50, color='white', zorder=5,
                           edgecolors=col, linewidths=1.8)

    custom_accs = [62.7,82.2,93.3,98.0,100,91.1]
    for i, acc in enumerate(custom_accs):
        ax.text(x[i], acc+1.5, f"{acc:.0f}%", ha='center', va='bottom',
                fontsize=7, color=ds_colors["Custom"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, fontweight='bold')
    ax.set_yticks([0,25,50,75,100])
    ax.set_yticklabels(["0%","25%","50%","75%","100%"], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlabel("Model (by parameter count)", fontsize=10)
    ax.set_ylim(30, 112)
    ax.set_xlim(-0.4, 5.6)

    legend_elements = [
        Line2D([0],[0], color="#1f77b4", lw=2, linestyle=(0,(3,1,1,1)), label='GSM8K'),
        Line2D([0],[0], color="#ff7f0e", lw=2, linestyle=(0,(5,3)),     label='AQuA-RAT'),
        Line2D([0],[0], color="#d62728", lw=2, linestyle='solid',       label='Custom dataset'),
        mpatches.Patch(color="#d62728", label='Filled circle = latent hallucination'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right')
    ax.set_title("Cross-Dataset Accuracy", fontsize=12,
                 fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart4_accuracy_comparison.png", dpi=180, bbox_inches='tight')
    print("Saved: chart4_accuracy_comparison.png")
    plt.close()


if __name__ == "__main__":
    chart1()
    chart2()
    chart3()
    chart4()
    print("\nAll charts saved.")
