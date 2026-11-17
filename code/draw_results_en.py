from figure_style import *
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

V=read('evaluation.json')
MAIN=list({x['semantic_program_sha256']:x for x in V['cohorts']['main144']['programs']}.values())
CONF=V['cohorts']['new_confirmation72']['programs']
SWEEP=V['cohorts']['sweep120']['programs']
PUBLIC=V['cohorts']['public24']['programs']
EVAL='data/evaluation.json'
assert (len(MAIN),len(CONF),len(SWEEP),len(PUBLIC))==(134,72,120,24)
COHORTS=[(MAIN,'Main',BLUE,'o'),(CONF,'Confirmation',TEAL,'^')]
PATTERNS=[('two','Two-vertex classes',BLUE,'o'),('mixed','Mixed sizes',ORANGE,'^')]
WEIGHTS=[('uniform','Uniform',TEAL,'o'),('periodic','Periodic',BLUE,'s'),('twolevel','Two-level',ORANGE,'^')]

def grid(height, rows=2, cols=2, left=.113, right=.977, bottom=.125, top=.91, wspace=.40, hspace=.72):
    f, aa=plt.subplots(rows,cols,figsize=(WIDTH,height),squeeze=False)
    f.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)
    return f,aa.ravel()

def graph_means(rows):
    return [(g[0],float(np.mean([saving(x) for x in g]))) for g in group(rows,'graph_sha256').values()]

def equal_line(ax,lo,hi,log=False):
    if log:ax.set_xscale('log');ax.set_yscale('log')
    ax.set(xlim=(lo,hi),ylim=(lo,hi))
    ax.plot([lo,hi],[lo,hi],c=SLATE,ls=(0,(4,3)),lw=.8,zorder=1)


def main_results():
    f,(a,b,c,d)=grid(4.23,left=.128,wspace=.49,hspace=.78)
    panels=[(a,'a','Paired complete-circuit CX'),(b,'b','Graph-level reduction'),(c,'c','Class size and quotient degree'),(d,'d','Paired target orders')]
    for ax,l,s in panels:panel(ax,l,s)
    all_rows=MAIN+CONF
    lo=min(x['ours_metrics']['cx'] for x in all_rows)*.82
    hi=max(x['reference_metric_envelope']['cx'] for x in all_rows)*1.13
    equal_line(a,lo,hi,True)
    for rows,label,color,marker in COHORTS:
        a.scatter([x['reference_metric_envelope']['cx'] for x in rows],[x['ours_metrics']['cx'] for x in rows],s=15,c=color,marker=marker,alpha=.61,linewidths=0,label=f'{label} {len(rows)}')
        g=graph_means(rows);assert len(g)==(62 if label=='Main' else 36)
        z=np.sort([q for _,q in g]);b.step(np.r_[0,z],np.r_[0,np.arange(1,len(z)+1)/len(z)],where='post',c=color,lw=1.45,label=f'{label} ({len(z)})')
        vals=[]
        for gg in group(rows,'graph_sha256').values():
            by=group(gg,'order');assert {'ascending','fixed_shuffle'}<=by.keys()
            vals.append([np.mean([saving(x) for x in by[k]]) for k in ['ascending','fixed_shuffle']])
        d.scatter(*np.asarray(vals).T,c=color,marker=marker,s=19,alpha=.65,linewidths=0)
    a.set(xlabel='Reference CX envelope',ylabel='Compiled CX count');a.legend(loc='upper left')
    b.set(xlabel='Mean CX reduction per graph (%)',ylabel='Cumulative fraction',xlim=(0,38),ylim=(0,1.04));b.legend(loc='upper left',fontsize=8.0,handlelength=1.2)
    g=graph_means(MAIN)+graph_means(CONF);degrees=[2,3,4,5,6,7,8]
    values=np.array([[np.mean([y for x,y in g if x['degree']==degree and x['pattern']==pattern]) for degree in degrees] for pattern,_,_,_ in PATTERNS]);assert np.all(np.isfinite(values)) and np.max(values)<40
    c.grid(False);c.pcolormesh(np.arange(8)-.5,np.arange(3)-.5,values,cmap='Blues',vmin=0,vmax=40,edgecolors='white',linewidth=.24);c.invert_yaxis()
    c.set_xticks(range(7),degrees);c.set_yticks([0,1],['Two-vertex','Mixed']);c.set_xlabel('Quotient-graph degree')
    for i in range(2):
        for j in range(7):c.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',fontsize=7.8,color='white' if values[i,j]>25 else INK)
    equal_line(d,0,40);d.set(xlabel='Ascending-order reduction (%)',ylabel='Pseudorandom-order reduction (%)')
    save(f,'05_main_distribution',[EVAL],{'main_unique_programs':134,'main_labeled_graphs':62,'confirmation_programs':72,'confirmation_nonisomorphic_graphs':36,'aggregate_saving_pct':[summed_saving(MAIN),summed_saving(CONF)],'degree_pattern_graph_mean_pct':values.tolist()})


def amortization():
    f,(a,b,c,d)=grid(4.34,left=.128,wspace=.62,hspace=.83)
    for ax,l,s in [(a,'a','Reduction by weight pattern'),(b,'b','Source-level savings (128 qubits)'),(c,'c','Cache break-even (6 qubits)'),(d,'d','Resource ratios by weight')]:panel(ax,l,s)
    for family,label,color,mark in WEIGHTS:
        ps=[1,2,4,8] if family=='uniform' else [2,4,8]
        vals=[[saving(x) for x in SWEEP if x['sweeps']==p and x['weight_family']==family] for p in ps]
        assert all(len(z)==12 for z in vals)
        means=np.array([np.mean(z) for z in vals]);lo=np.array([min(z) for z in vals]);hi=np.array([max(z) for z in vals])
        xx=np.log2(ps)+{'uniform':-.07,'periodic':0,'twolevel':.07}[family]
        a.errorbar(xx,means,yerr=[means-lo,hi-means],fmt=mark+'-',ms=4,c=color,lw=.9,elinewidth=.72,capsize=2,label=label)
    a.set(xticks=range(4),xticklabels=[1,2,4,8],xlabel='Mixer sweeps',ylabel='Paired CX reduction (%)',ylim=(0,21));a.legend(loc='lower right',fontsize=7.6)
    case='n128_d4_mixed_s0'; rr=V['cohorts']['sweep120']['all_stage_programs'];points={}
    for x in rr:
        if x['graph_id']==case and x['stage']=='raw':
            m=x['all_candidate_metrics'];points.setdefault(x['sweeps'],set()).add(m['ours_none']['cx']-m['ours_all']['cx'])
    assert all(z=={1088*p-256} for p,z in points.items())
    ps=sorted(points);p=np.linspace(min(ps),max(ps),80)
    b.plot(p,1088*p-256,c=BLUE,lw=1.1,ls='--',label=r'$1088p-256$')
    b.scatter(ps,[next(iter(points[i])) for i in ps],s=26,c=TEAL,zorder=3,label='Measured')
    b.set(xticks=ps,xlabel='Mixer sweeps $p$',ylabel='Source-level CX savings');b.legend(loc='upper left',fontsize=7.9)
    rows=read('threshold.json')['rows'];lookup={(x['p'],x['mode']):{r['stage']:r['metrics']['cx'] for r in x['rows']} for x in rows}
    ps=sorted({p for p,_ in lookup});raw=[lookup[p,'none']['raw']-lookup[p,'all']['raw'] for p in ps];post=[lookup[p,'none']['post']-lookup[p,'all']['post'] for p in ps]
    assert raw==[4*p-12 for p in ps]
    yzero(c);c.plot(ps,raw,'o-',ms=4,c=BLUE,lw=1,label='Source level');c.plot(ps,post,'s--',ms=4,c=TEAL,lw=1,label='Post-processed')
    c.scatter([3],[0],s=68,facecolors='none',edgecolors=ORANGE,lw=1.05,zorder=4)
    c.annotate('Break-even at $p=3$',xy=(3,0),xytext=(3.7,-6),fontsize=7.6,color=ORANGE,
               arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.65))
    c.set(xticks=ps,xlabel='Mixer sweeps $p$',ylabel='CX savings with full caching');c.legend(loc='upper left',fontsize=7.9)
    keys=['cx','two_qubit_depth','total_1q','parameterized_rz']
    for family,label,color,mark in WEIGHTS:
        ss=[x for x in SWEEP if x['weight_family']==family and x['sweeps']>=2]
        ratio=[sum(x['ours_metrics'][k] for x in ss)/sum(x['reference_CXwinner_metrics'][k] for x in ss) for k in keys]
        offset={'uniform':-.17,'periodic':0,'twolevel':.17}[family]
        d.scatter(ratio,np.arange(4)+offset,marker=mark,c=color,s=27,label=label)
    d.axvline(1,c=SLATE,lw=.8,ls='--');d.set_yticks(range(4),['CX','Two-qubit depth','1Q gates','Param. $R_z$']);d.invert_yaxis();d.set_xlabel('Ratio to CX-optimal reference')
    save(f,'08_amortization',[EVAL,'data/threshold.json'],{'p1_only_uniform':True,'panel_a_range':'observed min/max, 12 inputs per condition','large_raw_affine':'1088p-256','threshold_source':raw,'threshold_post':post,'shared_condition_in_d':'p >= 2'})



if __name__ == "__main__":
    main_results()
    amortization()
