from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='График динамики SOFA')
parser.add_argument('--data', type=Path, default=base/'data/sofa.csv')
parser.add_argument('--style', type=Path, default=base/'style.json')
parser.add_argument('--output', type=Path, default=base/'output')
args = parser.parse_args()
style = json.loads(args.style.read_text(encoding='utf-8'))
out = args.output
out.mkdir(parents=True, exist_ok=True)
with args.data.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise ValueError('Нет пациентов в таблице')
if any(r['survived_28d'] not in ('0', '1') for r in rows):
    raise ValueError('survived_28d должен быть 0 или 1')
x = np.array([0,24,48,72,120])
data = np.array([[float(r[f'sofa_{t}']) if r[f'sofa_{t}'].strip() else np.nan for t in x] for r in rows])
surv = np.array([r['survived_28d'] == '1' for r in rows])
if not np.all((data[np.isfinite(data)] >= 0) & (data[np.isfinite(data)] <= 24)):
    raise ValueError('SOFA должен быть в диапазоне 0–24')
if not np.all(np.isfinite(data[:,0])):
    raise ValueError('SOFA 0 должен быть указан для каждого пациента')
if not surv.any() or surv.all():
    raise ValueError('Для двухпанельного графика нужны оба 28-дневных исхода')
plt.rcParams.update({'font.family':style['font_family'],'font.size':style['font_size'],'svg.fonttype':'none','axes.labelcolor':'#263548','text.color':'#263548','xtick.color':'#263548','ytick.color':'#263548'})
fig,axs=plt.subplots(1,2,figsize=style['figure_size'])
fig.subplots_adjust(left=.065,right=.975,bottom=.30,top=.80,wspace=.18)
fig.suptitle(style['title'],fontsize=style['title_size'],fontweight='bold',y=.96)
fig.text(.5,.895,f'{len(rows)} пациентов · 28-дневный исход: {surv.sum()} выживших, {(~surv).sum()} умерших',ha='center',fontsize=14,color='#566578')
blue=style['colors']['median']; green=style['colors']['survivors']; red=style['colors']['deceased']
def stats(a):
    return np.nanquantile(a,[.25,.5,.75],axis=0,method='linear')
def setup(ax,title):
    ax.set_title(title,loc='left',fontweight='bold',fontsize=16,pad=18)
    ax.set_xlim(-6,126);ax.set_ylim(0,24)
    ax.set_xticks(x);ax.set_yticks(np.arange(0,25,4))
    ax.set_xlabel('Часы от первой гемосорбции',labelpad=12)
    ax.set_ylabel('SOFA, баллы',labelpad=8)
    ax.grid(axis='y',color='#E7EBF0',linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#A9B3BF')
def tracks(ax,a,color,alpha):
    for y in a:
        ax.plot(x,y,color=color,alpha=alpha,lw=style['patient_line_width'],marker='o',ms=style['patient_marker_size'],zorder=2)
        if np.sum(np.isfinite(y))==1:
            ax.scatter(x[np.isfinite(y)],y[np.isfinite(y)],marker='D',s=55,facecolor='white',edgecolor=color,linewidth=1.8,zorder=6)
def summary(ax,a,color):
    q=stats(a)
    ax.fill_between(x,q[0],q[2],color=color,alpha=style['quartile_alpha'],zorder=1)
    ax.plot(x,q[1],color=color,lw=style['median_line_width'],marker='o',ms=style['median_marker_size'],zorder=5)
def ns(ax,a,y,color,label):
    ax.text(-.01,y,label,transform=ax.transAxes,ha='right',va='center',fontsize=11,color=color)
    for t,n in zip(x,np.sum(np.isfinite(a),axis=0)):
        ax.text(t,y,str(n),transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=11,color=color)
setup(axs[0],'А. Все индивидуальные траектории')
tracks(axs[0],data,'#8C98A7',.48);summary(axs[0],data,blue)
axs[0].legend(handles=[Line2D([0],[0],color=blue,lw=3,marker='o',label='Медиана'),Patch(facecolor=blue,alpha=.13,label='25–75-й процентили'),Line2D([0],[0],color='#8C98A7',lw=1.1,label='Отдельный пациент'),Line2D([0],[0],marker='D',mfc='white',mec='#8C98A7',ls='',label='Только SOFA 0')],loc='lower left',frameon=False,fontsize=11)
ns(axs[0],data,-.24,'#566578','n')
setup(axs[1],'Б. Траектории по 28-дневному исходу')
for mask,color in [(~surv,red),(surv,green)]:
    tracks(axs[1],data[mask],color,.28);summary(axs[1],data[mask],color)
axs[1].legend(handles=[Line2D([0],[0],color=green,lw=3,marker='o',label='Выжившие к 28-м суткам'),Line2D([0],[0],color=red,lw=3,marker='o',label='Умершие к 28-м суткам'),Patch(facecolor='#8595A8',alpha=.16,label='25–75-й процентили')],loc='lower left',frameon=False,fontsize=11)
ns(axs[1],data[surv],-.24,green,'n выживших')
ns(axs[1],data[~surv],-.315,red,'n умерших')
fig.text(.065,.065,'Тонкие линии — отдельные пациенты; толстые — медианы; заливка — межквартильный диапазон.\nВ каждой точке учтены только доступные измерения; отсутствующие значения не восстанавливались.',fontsize=11,color='#566578',linespacing=1.6)
fig.savefig(out/'SOFA_LPS_GEMA.png',dpi=style['png_dpi'],facecolor='white')
fig.savefig(out/'SOFA_LPS_GEMA.svg',facecolor='white')
plt.close(fig)
for label,a in [('all',data),('survivors',data[surv]),('deceased',data[~surv])]:
    print(label,'n=',np.sum(np.isfinite(a),axis=0).tolist(),'quartiles=',stats(a).tolist())
