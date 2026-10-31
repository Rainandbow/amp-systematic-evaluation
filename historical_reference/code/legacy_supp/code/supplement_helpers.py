"""Portable style, frozen estimates and blocking QA for all supplementary figures."""
from pathlib import Path
import csv,hashlib,json,math,sys,subprocess
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
OUT=None
D=json.loads((ROOT/'data/frozen_estimates.json').read_text(encoding='utf-8'))
DS=D['datasets'];MS=D['tools']
NAMES={'tsAMP_I':'tsAMP-I','HMD_AMP':'HMD-AMP','c_AMPs_benchmark_resplit':'c_AMPs','AmPEP_raw_source':'AmPEP','ampir_raw_mature90':'ampir'}
COLORS=['#0072B2','#00875A','#A66A00','#AA62A0','#407DB1','#D55E00'];MARKERS=['o','s','^','D','P','X']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6.5,'legend.fontsize':6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.facecolor':'white'})
def name(x):return NAMES.get(x,x)
def one(rows,**kw):
 matches=[r for r in rows if all(r.get(k)==v for k,v in kw.items())]
 assert len(matches)==1,kw
 return matches[0]
def num(x):
 try:v=float(x)
 except (TypeError,ValueError):return None
 return v if math.isfinite(v) else None
def read_tsv(path):
 path=Path(path)
 if not path.exists() and path.parent.name=='source_data':path=ROOT/'data'/path.name
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def style(ax,grid='x'):
 for spine in ax.spines.values():spine.set_color('#9eabb5')
 ax.tick_params(width=.5,length=2.5,color='#9eabb5')
 if grid:ax.grid(axis=grid,color='#e7edf1',lw=.45);ax.set_axisbelow(True)
def heading(ax,letter,title):
 trans=ax.transAxes+ScaledTranslation(0,10/72,ax.figure.dpi_scale_trans)
 ax.text(0,1,letter+'  '+title,transform=trans,fontsize=7,va='bottom')
def legend(fig,n=6,y=.99):
 fig.legend(handles=[Line2D([],[],color=COLORS[i],marker=MARKERS[i],lw=.7,ms=3,label=name(m)) for i,m in enumerate(MS[:n])],loc='upper center',bbox_to_anchor=(.53,y),ncol=3 if n==6 else 5,fontsize=6.2,columnspacing=1.4,handletextpad=.4)
def save(fig,stem,axes,ids=None,rows=None,cols=None):
 assert OUT is not None
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids or list('abcdef')[:len(axes)],row_groups=rows,column_groups=cols,json_out=OUT/'QA'/(stem+'.alignment.json'),strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
 fig.savefig(OUT/'PDF'/(stem+'.pdf'),dpi=600)
 fig.savefig(OUT/'SVG'/(stem+'.svg'),dpi=600)
 fig.savefig(OUT/'PNG_600dpi'/(stem+'.png'),dpi=600)
 fig.savefig(OUT/'TIFF_600dpi'/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(OUT/'preview'/(stem+'.png'),dpi=300);plt.close(fig)
 for script,extra in [('audit_pdf_text.py',[]),('audit_figure_collisions.py',['--json-out',str(OUT/'QA'/(stem+'.collision.json')),'--strict'])]:
  r=subprocess.run([sys.executable,str(ROOT/'code'/script),str(OUT/'PDF'/(stem+'.pdf')),*extra],capture_output=True,text=True,encoding='utf-8',errors='replace')
  (OUT/'QA'/(stem+'.'+script+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
  if r.returncode:raise RuntimeError('Figure QA failed: '+stem+'; see '+script+'.log')
 print('FIGURE PASS',stem,flush=True)
