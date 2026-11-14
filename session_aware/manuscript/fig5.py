from figstyle import *
import pandas as pd, numpy as np
Dz=pd.read_csv(DAT+'ew_disease.csv'); B=pd.read_csv(DAT+'fig3_dose_response.csv')
fig,ax=plt.subplots(1,2,figsize=(5.6,2.6),gridspec_kw=dict(wspace=0.40,left=0.10,right=0.985,bottom=0.19,top=0.90,width_ratios=[1.35,1]))
a=ax[0]; order=['(0, 68]','(68, 72]','(72, 75]','(75, 77]','(77, 79]','(79, 81]','(81, 90]']; labs=['≤68','68–72','72–75','75–77','77–79','79–81','>81']; x=np.arange(7)
a.axhline(0,color=DK,lw=0.8)
for out,col,off,lab in (('drop',RED,-0.1,'Cows in acute drop'),('onset',BLUE,0.1,'Onsets among cows at risk')):
    r=B[(B.outcome==out)&(B.lag=='tmax_l1')].set_index('b').reindex(order); a.errorbar(x+off,r['mean']*100,yerr=r['sem']*196,fmt='o',ms=3.5,color=col,elinewidth=0.8,capsize=0,label=lab)
a.set_xticks(x); a.set_xticklabels(labs,fontsize=6.3); a.set_xlabel('Daily maximum THI, 1 day earlier'); a.set_ylabel('Difference (percentage points)'); a.legend(loc='upper left',handletextpad=0.2); panel(a,'a',x=-0.2,y=1.03)
b=ax[1]; d=Dz[Dz.config.isin(['R_deviation','M_ref'])].dropna(subset=['hot_sens10','mild_sens10'])
for cfg,col,lab in (('R_deviation',DK,'Deviation rule'),('M_ref',RED,'Main model')):
    z=d[d.config==cfg]
    for _,r in z.iterrows(): b.plot([0,1],[r.mild_sens10*100,r.hot_sens10*100],color=col,lw=0.6,alpha=0.55)
    b.plot([0,1],[z.mild_sens10.mean()*100,z.hot_sens10.mean()*100],color=col,lw=2.0,marker='o',ms=4,label=lab)
b.set_xticks([0,1]); b.set_xticklabels(['≤72','≥77']); b.set_xlabel('3-day mean of daily maximum THI'); b.set_xlim(-0.35,1.35); b.set_ylabel('Cow-days on 10% lists (%)'); b.set_ylim(None,82); b.legend(loc='upper left',handletextpad=0.3,fontsize=6.5,borderaxespad=0.2); panel(b,'b',x=-0.28,y=1.03)
save(fig,'Fig5')
