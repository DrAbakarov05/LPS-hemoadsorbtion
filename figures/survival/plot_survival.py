"""Build separate 28-day Kaplan–Meier curves from anonymized follow-up times."""
from pathlib import Path
import csv, math, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
OUTPUT=ROOT/'output'
OUTPUT.mkdir(exist_ok=True)
groups=[]
for slug,color in [('LPS_GEMA','#176b87'),('Efferon_LPS','#7847a3')]:
    with (ROOT/'data'/f'{slug}_survival.csv').open(encoding='utf-8-sig') as f:
        data=list(csv.DictReader(f))
    rows=[]
    for r in data:
        duration=int(r['time_days']); event=int(r['event']); survived=int(r['survived_28d'])
        if not 0<=duration<=28 or event not in (0,1) or survived!=1-event:
            raise ValueError(f"Invalid survival record: {r['record_id']}")
        if not event and duration!=28:
            raise ValueError('This dataset requires confirmed 28-day follow-up for survivors')
        rows.append({'id':r['record_id'],'duration':duration,'event':event,'included':1})
    included=[r for r in rows if r['included']]
    n=len(included); d=sum(r['event'] for r in included)
    times=np.array([r['duration'] for r in included])
    events=np.array([r['event'] for r in included])
    event_times=sorted(int(t) for t in set(times[events==1]))
    x=[0.]; y=[1.]; lo=[1.]; hi=[1.]; s=1.; g=0.; table=[]
    for t in event_times:
        risk=int(sum(times>=t)); deaths=int(sum((times==t)&(events==1)))
        s*=1-deaths/risk
        if risk>deaths: g+=deaths/(risk*(risk-deaths))
        if 0<s<1:
            se=math.sqrt(g)/abs(math.log(s)); z=math.log(-math.log(s))
            l=math.exp(-math.exp(z+1.95996398454*se)); h=math.exp(-math.exp(z-1.95996398454*se))
        else: l=h=s
        x.append(float(t)); y.append(s); lo.append(l); hi.append(h)
        table.append({'day':int(t),'at_risk':risk,'deaths':deaths,'survival':s,'lower95':l,'upper95':h})
    x.append(28); y.append(s); lo.append(lo[-1]); hi.append(hi[-1])
    fig=plt.figure(figsize=(8,6.5))
    ax=fig.add_axes([.13,.26,.82,.60])
    ax.step(x,y,where='post',color=color,lw=2.6,label='Каплан—Майер')
    ax.fill_between(x,lo,hi,step='post',color=color,alpha=.14,label='95% ДИ')
    ax.plot([28],[s],marker='+',markersize=12,color=color,mew=2)
    ax.text(27.3,s+.045,f'{s:.1%}'.replace('.',','),ha='right',color=color,weight='bold')
    ax.set(xlim=(0,28.7),ylim=(0,1.03),xlabel='Дни от первой процедуры',ylabel='Вероятность выживания')
    ax.set_xticks([0,7,14,21,28]); ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis='y',alpha=.18); ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper right',frameon=False,fontsize=10)
    title='LPS-GEMA' if slug=='LPS_GEMA' else 'Эфферон LPS'
    fig.text(.13,.94,title,fontsize=19,weight='bold')
    rax=fig.add_axes([.13,.07,.82,.065])
    rax.set_xlim(0,28.7); rax.set_ylim(0,1); rax.axis('off')
    fig.text(.13,.15,'Число под наблюдением',fontsize=11,color='#555555')
    for t in [0,7,14,21,28]:
        rax.text(t,.5,str(int(sum(times>=t))),ha='center',va='center')
    full_deaths=sum(r['event'] for r in rows)
    for ext in ['png','svg','pdf']:
        fig.savefig(OUTPUT/(slug+'_KM_28d.'+ext),dpi=600,transparent=True)
    plt.close(fig)
    with (OUTPUT/(slug+'_KM_table.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
    assert abs(s-(n-d)/n)<1e-12
    groups.append({'group':title,'total_n':len(rows),'deaths_28':full_deaths,'KM_n':n,'KM_deaths':d,'KM_survival28':s,'CI95':[lo[-1],hi[-1]],'event_days':event_times})
(OUTPUT/'summary.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
print(json.dumps(groups,ensure_ascii=False,indent=2))
