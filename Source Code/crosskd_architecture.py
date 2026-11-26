import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

def box(ax, x, y, w, h, text, color='#4A90D9', fontsize=9, textcolor='white', style='round,pad=0.1'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=style,
                          linewidth=1.5, edgecolor=color,
                          facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor,
            wrap=True, multialignment='center')

def arrow(ax, x1, y1, x2, y2, color='black', lw=1.5, style='->', label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=7.5, color=color, va='center')

# ── Judul ─────────────────────────────────────────────────────────────
ax.text(7, 7.6, 'Arsitektur Knowledge Distillation (CrossKD)',
        ha='center', va='center', fontsize=13, fontweight='bold')

# ── Input ─────────────────────────────────────────────────────────────
box(ax, 5.75, 6.6, 2.5, 0.6, 'Input Citra (640x640)', color='#555555', fontsize=9)

# ── Teacher ───────────────────────────────────────────────────────────
box(ax, 0.3, 4.8, 2.8, 0.65, 'Teacher Backbone\n(YOLOv8l)', color='#C0392B', fontsize=8.5)
box(ax, 0.3, 3.7, 2.8, 0.65, 'Teacher Neck (FPN)', color='#E74C3C', fontsize=8.5)
box(ax, 0.3, 2.6, 2.8, 0.65, 'Teacher Decoder\n(Detect Head)', color='#E74C3C', fontsize=8.5)
box(ax, 0.3, 1.5, 2.8, 0.65, 'Teacher Output\n(cls score, bbox)', color='#922B21', fontsize=8.5)

# ── Student ───────────────────────────────────────────────────────────
box(ax, 10.9, 4.8, 2.8, 0.65, 'Student Backbone\n(YOLOv8n)', color='#1A5276', fontsize=8.5)
box(ax, 10.9, 3.7, 2.8, 0.65, 'Student Neck (FPN)', color='#2980B9', fontsize=8.5)
box(ax, 10.9, 2.6, 2.8, 0.65, 'Student Decoder\n(Detect Head)', color='#2980B9', fontsize=8.5)
box(ax, 10.9, 1.5, 2.8, 0.65, 'Student Output\n(cls score, bbox)', color='#1A5276', fontsize=8.5)

# ── Loss boxes ────────────────────────────────────────────────────────
box(ax, 4.5, 0.3, 2.2, 0.65, 'Distillation Loss\n(CrossKD)', color='#8E44AD', fontsize=8.5)
box(ax, 7.3, 0.3, 2.2, 0.65, 'Task Loss\n(box+cls+dfl)', color='#27AE60', fontsize=8.5)
box(ax, 5.75, 6.0, 2.5, 0.5, 'Total Loss = Task Loss\n+ Distillation Loss', color='#2C3E50', fontsize=8)

# ── Ground Truth ──────────────────────────────────────────────────────
box(ax, 9.5, 1.5, 1.2, 0.65, 'Ground\nTruth', color='#27AE60', fontsize=8)

# ── Arrows input -> backbone ──────────────────────────────────────────
arrow(ax, 5.75, 6.6, 3.1, 5.45, color='#C0392B', lw=1.5)   # -> teacher
arrow(ax, 8.25, 6.6, 10.9, 5.45, color='#1A5276', lw=1.5)  # -> student

# ── Arrows teacher chain ──────────────────────────────────────────────
arrow(ax, 1.7, 4.8, 1.7, 4.35, color='#C0392B')
arrow(ax, 1.7, 3.7, 1.7, 3.25, color='#C0392B')
arrow(ax, 1.7, 2.6, 1.7, 2.15, color='#C0392B')

# ── Arrows student chain ──────────────────────────────────────────────
arrow(ax, 12.3, 4.8, 12.3, 4.35, color='#1A5276')
arrow(ax, 12.3, 3.7, 12.3, 3.25, color='#1A5276')
arrow(ax, 12.3, 2.6, 12.3, 2.15, color='#1A5276')

# ── CrossKD: teacher output -> distillation loss ──────────────────────
arrow(ax, 1.7, 1.5, 5.0, 0.95, color='#8E44AD', lw=1.5, label='cls_kd + reg_kd')

# ── Student output -> distillation loss ──────────────────────────────
arrow(ax, 11.5, 1.5, 6.7, 0.95, color='#8E44AD', lw=1.5)

# ── Student output -> task loss ───────────────────────────────────────
arrow(ax, 12.3, 1.5, 8.4, 0.95, color='#27AE60', lw=1.5)

# ── Ground truth -> task loss ─────────────────────────────────────────
arrow(ax, 10.1, 1.825, 9.5, 0.95, color='#27AE60', lw=1.5)

# ── Both losses -> total loss ─────────────────────────────────────────
arrow(ax, 5.6, 0.95, 6.5, 6.0, color='#2C3E50', lw=1.5)
arrow(ax, 8.4, 0.95, 7.5, 6.0, color='#2C3E50', lw=1.5)

# ── Teacher frozen label ──────────────────────────────────────────────
ax.text(1.7, 5.6, 'FROZEN\n(tidak diupdate)', ha='center', va='center',
        fontsize=7.5, color='#C0392B', style='italic')
ax.text(12.3, 5.6, 'TRAINABLE\n(diupdate)', ha='center', va='center',
        fontsize=7.5, color='#1A5276', style='italic')

# ── Legend ────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color='#C0392B', label='Teacher Model (YOLOv8l)'),
    mpatches.Patch(color='#2980B9', label='Student Model (YOLOv8n)'),
    mpatches.Patch(color='#8E44AD', label='Distillation Loss (CrossKD)'),
    mpatches.Patch(color='#27AE60', label='Task Loss'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=4,
          fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('crosskd_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: crosskd_architecture.png")
