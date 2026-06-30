#!/usr/bin/env python3
"""
cpg_architecture_diagram.py
Generates the *as-implemented* architecture of one leg of the tinyCPG model
(the contralateral leg is the mirror image; the commissural link is shown as
a stub). Drawn directly from the projection list verified by
--dump-connectivity, so it reflects what the code actually builds, not an
idealised schematic.

Colour key (bio-meaningful):
  excitatory neurons      -> orange nodes
  excitatory projections  -> light orange (static) / dark orange (STDP/plastic)
  inhibitory neurons+proj -> light blue
  muscles                 -> red
  inputs/afferents        -> grey   (rate-coded transduction = grey dashed)
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

EXC_P="#C2570B"   # STDP / plastic excitatory projection (dark orange)
EXC_S="#F2B66D"   # static excitatory projection (light orange)
INH="#4FA8D8"     # inhibitory neuron + projection (light blue)
RATE="#9e9e9e"    # rate-coded transduction / input grey

def node(ax, xy, label, kind="rg", r=0.42):
    x,y=xy
    fc={"rg":"#F59331",     # excitatory neuron (orange)
        "m":"#F59331",      # excitatory motoneuron (orange)
        "in":"#9FD3EC",     # inhibitory interneuron (light blue)
        "iaint":"#9FD3EC",  # Ia inhibitory interneuron (light blue)
        "mus":"#D7263D",    # muscle (red)
        "drive":"#BdBdBd",  # input / descending drive (grey)
        "aff":"#BdBdBd"}[kind]  # afferent input (grey)
    if kind in ("mus","drive"):
        b=FancyBboxPatch((x-0.62,y-0.28),1.24,0.56,boxstyle="round,pad=0.02",
                         fc=fc,ec="#333",lw=1.2,zorder=3)
        ax.add_patch(b)
    else:
        ax.add_patch(Circle((x,y),r,fc=fc,ec="#333",lw=1.2,zorder=3))
    ax.text(x,y,label,ha="center",va="center",fontsize=8.5,fontweight="bold",zorder=4)

def edge(ax, a, b, color, style="-", rad=0.0, lw=1.6, label=None, lx=0.5):
    ar=FancyArrowPatch(a,b,connectionstyle=f"arc3,rad={rad}",arrowstyle="-|>",
                       mutation_scale=13,lw=lw,color=color,linestyle=style,
                       shrinkA=16,shrinkB=16,zorder=2)
    ax.add_patch(ar)
    if label:
        mx=a[0]+(b[0]-a[0])*lx+rad*1.2; my=a[1]+(b[1]-a[1])*lx+0.18
        ax.text(mx,my,label,fontsize=6.6,color=color,ha="center",style="italic",zorder=5)

fig,ax=plt.subplots(figsize=(15,8.6))
ax.set_xlim(0,15.5); ax.set_ylim(0,10); ax.axis("off")

# positions
P=dict(BS=(1,9), CUT=(1,7.2), base=(1,5.6),
       IaEext=(1,3.4), flexAff=(1,1.6),
       RGE=(4.3,8.4), InE=(4.3,6.6), InF=(4.3,4.0), RGF=(4.3,2.2),
       IaIntE=(6.5,7.7), IaIntF=(6.5,2.9),
       ME=(8.7,8.4), MF=(8.7,2.2),
       musE=(11,8.4), musF=(11,2.2),
       IaE=(13.4,6.6), IaF=(13.4,4.0))
for k,(lab,kind) in {
    "BS":("BS","drive"),"CUT":("CUT","drive"),"base":("base","drive"),
    "IaEext":("Ia-E\nramp","aff"),"flexAff":("flex\nAff","aff"),
    "RGE":("RG-E","rg"),"RGF":("RG-F","rg"),"InE":("InE","in"),"InF":("InF","in"),
    "IaIntE":("IaInt-E","iaint"),"IaIntF":("IaInt-F","iaint"),
    "ME":("M-E","m"),"MF":("M-F","m"),"musE":("mus-E","mus"),"musF":("mus-F","mus"),
    "IaE":("Ia-E","aff"),"IaF":("Ia-F","aff")}.items():
    node(ax,P[k],lab,kind)

# --- descending / supraspinal ---
edge(ax,P["BS"],P["RGE"],EXC_S,label="BS→E (frozen)",lx=0.42)
edge(ax,P["BS"],P["RGF"],EXC_S,rad=-0.25,label="BS→F (frozen)",lx=0.2)
edge(ax,P["CUT"],P["RGE"],EXC_P,rad=0.12,label="plastic →63",lx=0.55)
edge(ax,P["CUT"],P["InE"],EXC_S,rad=-0.1,label="cut reflex",lx=0.7)
edge(ax,P["base"],P["RGE"],EXC_S,rad=0.18,lw=1.0)
edge(ax,P["base"],P["RGF"],EXC_S,rad=-0.18,lw=1.0,label="tonic bias",lx=0.3)
# --- external pacing afferents ---
edge(ax,P["IaEext"],P["RGE"],EXC_S,rad=0.3,style=":",label="heel→toe ramp",lx=0.25)
edge(ax,P["flexAff"],P["RGF"],EXC_S,style=":",label="swing aff",lx=0.5)
edge(ax,P["flexAff"],P["InF"],EXC_S,rad=0.15,lw=1.0)
# --- RG reciprocal core ---
edge(ax,P["RGE"],P["InE"],EXC_S,rad=0.0)
edge(ax,P["RGF"],P["InF"],EXC_S,rad=0.0)
edge(ax,P["InE"],P["RGF"],INH,rad=0.25,label="−8",lx=0.5)
edge(ax,P["InF"],P["RGE"],INH,rad=-0.25,label="−48 (Zhang 6:1)",lx=0.5)
# --- motor output ---
edge(ax,P["RGE"],P["ME"],EXC_S,label="+30",lx=0.5)
edge(ax,P["RGF"],P["MF"],EXC_S,label="+30",lx=0.5)
edge(ax,P["ME"],P["MF"],INH,rad=0.35,lw=1.1,label="motor recip −22",lx=0.5)
edge(ax,P["MF"],P["ME"],INH,rad=0.35,lw=1.1)
edge(ax,P["ME"],P["musE"],EXC_S)
edge(ax,P["MF"],P["musF"],EXC_S)
# --- sensory feedback loop (rate-coded) ---
edge(ax,P["musE"],P["IaE"],RATE,style="--",label="force/len→Hz",lx=0.5)
edge(ax,P["musF"],P["IaF"],RATE,style="--")
# Ia interneuron pathway
edge(ax,P["IaE"],P["IaIntE"],EXC_S,rad=0.1,lw=1.1)
edge(ax,P["IaIntE"],P["MF"],INH,rad=0.0,lw=1.1,label="Ia recip −10",lx=0.6)
edge(ax,P["IaF"],P["IaIntF"],EXC_S,rad=-0.1,lw=1.1)
edge(ax,P["IaIntF"],P["ME"],INH,rad=0.0,lw=1.1)
# Ia closed loop into RG interneurons
edge(ax,P["IaE"],P["InE"],EXC_S,rad=0.3,lw=1.1,style="-")
edge(ax,P["IaF"],P["InF"],EXC_S,rad=-0.3,lw=1.1,label="Ia→In loop",lx=0.4)
# Ia plastic into RG (sensory arm)
edge(ax,P["IaE"],P["RGE"],EXC_P,rad=0.45,lw=2.0,label="plastic Ia→RG (sensory)",lx=0.3)
edge(ax,P["IaF"],P["RGF"],EXC_P,rad=-0.45,lw=2.0)
# commissural stub
ax.annotate("commissural\nRG-E↔RG-E (−8)\nRG-F↔RG-F (−20)\n→ contralateral leg",
            (4.3,0.5),fontsize=7.5,color=INH,ha="center",
            bbox=dict(boxstyle="round",fc="#fff3f3",ec=INH,lw=1))
edge(ax,P["RGF"],(4.3,1.05),INH,rad=0.0,lw=1.2)

# legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
leg=[Patch(fc="#F59331",ec="#333",label="excitatory neuron"),
     Patch(fc="#9FD3EC",ec="#333",label="inhibitory interneuron"),
     Patch(fc="#D7263D",ec="#333",label="muscle"),
     Patch(fc="#BdBdBd",ec="#333",label="input / afferent"),
     Line2D([0],[0],color=EXC_P,lw=2.5,label="plastic excitatory (STDP)"),
     Line2D([0],[0],color=EXC_S,lw=2.5,label="static excitatory"),
     Line2D([0],[0],color=INH,lw=2.5,label="inhibitory projection"),
     Line2D([0],[0],color=RATE,lw=2.5,ls="--",label="rate-coded transduction"),
     Line2D([0],[0],color=EXC_S,lw=2.5,ls=":",label="external pacing afferent")]
ax.legend(handles=leg,loc="upper center",ncol=5,fontsize=8.2,frameon=True,bbox_to_anchor=(0.5,1.05))
ax.set_title("tinyCPG — as-implemented architecture (one leg; contralateral is mirror)",
             fontsize=13,fontweight="bold",y=1.06)
fig.savefig("plots/paper/fig_architecture_implemented.png",dpi=160,bbox_inches="tight")
print("saved plots/paper/fig_architecture_implemented.png")
