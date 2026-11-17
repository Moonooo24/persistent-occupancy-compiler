from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'paper/figures'
OUT.mkdir(parents=True, exist_ok=True)
for p in [ROOT/'tmp/matplotlib', ROOT/'tmp']:
    p.mkdir(parents=True, exist_ok=True)
os.environ.update(MPLCONFIGDIR=str(ROOT/'tmp/matplotlib'), TEMP=str(ROOT/'tmp'), TMP=str(ROOT/'tmp'))
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt, font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np

FONT_DIR = Path(os.environ.get('JOURNAL_FONT_DIR', 'C:/Windows/Fonts'))
for name in ['times.ttf','timesi.ttf','timesbd.ttf','timesbi.ttf']:
    path = FONT_DIR/name
    if not path.is_file():
        raise FileNotFoundError(f'Required figure font: {path}')
    fm.fontManager.addfont(path)
TIMES = 'Times New Roman'
fm.findfont(fm.FontProperties(family=TIMES), fallback_to_default=False)
INK = '#1E2D39'
SLATE = '#526675'
TEAL = '#007C76'
BLUE = '#305C92'
ORANGE = '#A55A1A'
PURPLE = '#674D91'
GRID = '#E0E7EB'
PALE = '#F3F7FA'
PALE_T = '#E8F4F0'
PALE_P = '#F1EDF8'
PALE_O = '#FAF0E3'
RANK_COLORS = {2: BLUE, 4: TEAL, 6: ORANGE, 8: PURPLE}
WIDTH = 6.30
mpl.rcParams.update({
    'font.family': TIMES, 'font.size': 9.0,
    'mathtext.fontset': 'custom', 'mathtext.rm': TIMES,
    'mathtext.it': TIMES+':italic', 'mathtext.bf': TIMES+':bold',
    'mathtext.bfit': TIMES+':bold:italic', 'mathtext.sf': TIMES,
    'mathtext.tt': TIMES, 'mathtext.cal': TIMES+':italic', 'mathtext.fallback': None,
    'pdf.fonttype':42, 'pdf.use14corefonts':False, 'svg.fonttype':'none',
    'figure.facecolor':'white', 'savefig.facecolor':'white',
    'text.color':INK, 'axes.labelcolor':INK, 'axes.edgecolor':SLATE,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
    'axes.grid':True, 'axes.axisbelow':True, 'grid.color':GRID, 'grid.linewidth':.55,
    'xtick.color':SLATE, 'ytick.color':SLATE, 'xtick.labelsize':8.2, 'ytick.labelsize':8.2,
    'xtick.major.size':2.6, 'ytick.major.size':2.6, 'legend.frameon':False,
    'legend.fontsize':8.3, 'legend.handlelength':1.5, 'axes.labelsize':8.7,
    'axes.unicode_minus':True,
})

def read(name):
    return json.loads((DATA/name).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(fig, name, sources, details):
    report_text_layout(fig, name)
    fig.savefig(OUT/f'{name}.pdf', metadata={'Title':name, 'Creator':'Matplotlib'})
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png', dpi=320)
    doc = dict(figure=name, sources=[dict(path=s, sha256=sha(ROOT/s)) for s in sources],
               details=details, width_mm=fig.get_figwidth()*25.4, height_mm=fig.get_figheight()*25.4)
    (OUT/f'{name}.provenance.json').write_text(json.dumps(doc, ensure_ascii=False,indent=2)+'\n', encoding='utf8')
    plt.close(fig)
    print(name, flush=True)

def canvas(height):
    f = plt.figure(figsize=(WIDTH,height))
    a = f.add_axes([0,0,1,1], xlim=(0,100), ylim=(0,100))
    a.axis('off')
    return f,a

def txt(a,x,y,s,size=9, color=INK, bold=False, **kw):
    return a.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',va='center',**kw)

def tag(a,x,y,l,s):
    txt(a,x,y,f'({l})',10,TEAL,True)
    txt(a,x+5.0,y,s,10,INK,True)

def panel(ax, letter, title):
    ax.set_title(f'({letter})  {title}', loc='left', fontsize=9.8, fontweight='bold', pad=7)

def rect(a,x,y,w,h,fill=PALE,edge=GRID,lw=.75,rounding=0):
    p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',fc=fill,ec=edge,lw=lw) if rounding else Rectangle((x,y),w,h,fc=fill,ec=edge,lw=lw)
    a.add_patch(p)
    return p

def arrow(a,start,end,color=SLATE,scale=8,**kw):
    a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=scale,lw=.85,color=color,shrinkA=0,shrinkB=0,**kw))

def wire(a,y,x1,x2,color=SLATE):
    a.plot([x1,x2],[y,y],color=color,lw=.8,zorder=1)

def dot(a,x,y,color=INK,size=13):
    a.scatter([x],[y],s=size,c=color,zorder=5,clip_on=False)

def gate(a,x,y,w,h,label,color=PURPLE,fill=PALE_P,size=9):
    rect(a,x-w/2,y-h/2,w,h,fill,color,rounding=.7).set_zorder(3)
    txt(a,x,y,label,size,color,ha='center',zorder=4)

def cx(a,x,control,target):
    a.plot([x,x],[control,target],color=PURPLE,lw=.85,zorder=2)
    dot(a,x,control,PURPLE,12)
    a.scatter(x,target,marker='o',s=38,facecolors='white',edgecolors=PURPLE,linewidths=.8,zorder=4)
    a.scatter(x,target,marker='+',s=27,c=PURPLE,linewidths=.8,zorder=5)

def yzero(ax):
    ax.axhline(0,c=SLATE,lw=.75,ls=(0,(3,3)),zorder=1)

def group(rows, key):
    out = {}
    for row in rows:out.setdefault(row[key],[]).append(row)
    return out

def saving(row):
    return 100*(1-row['ours_metrics']['cx']/row['reference_metric_envelope']['cx'])

def summed_saving(rows):
    return 100*(1-sum(x['ours_metrics']['cx'] for x in rows)/sum(x['reference_metric_envelope']['cx'] for x in rows))


def report_text_layout(fig, name):
    from matplotlib.text import Text
    from itertools import combinations

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    legends = list(fig.legends)
    for ax in fig.axes:
        artists.extend([ax.title, ax._left_title, ax._right_title, *ax.texts])
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
        for axis in (ax.xaxis, ax.yaxis):
            artists.extend([axis.label, axis.offsetText])
            low, high = sorted(axis.get_view_interval())
            artists.extend(a for x, a in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                           if low - 1e-8 <= x <= high + 1e-8)
    for legend in legends:
        artists.extend(legend.get_texts())
    boxes = [(a.get_text(), Text.get_window_extent(a, renderer))
             for a in artists if a.get_visible() and a.get_text().strip()]
    def intersect(a, b):
        return (min(a.x1,b.x1)-max(a.x0,b.x0), min(a.y1,b.y1)-max(a.y0,b.y0))
    for title,a in boxes:
        if a.x0 < -.25 or a.y0 < -.25 or a.x1 > fig.bbox.width+.25 or a.y1 > fig.bbox.height+.25:
            print("outside", name, title, [round(t,1) for t in a.extents])
    for (ta,a),(tb,b) in combinations(boxes,2):
        w,h=intersect(a,b)
        if w>1.4 and h>1.4:
            print("overlap",name,repr(ta),repr(tb),round(w,1),round(h,1))
