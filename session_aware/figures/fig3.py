from figstyle import *
import pandas as pd, numpy as np
B=pd.read_csv(DAT+'fig3_dose_response.csv'); PF=pd.read_csv(DAT+'fig3_per_farm.csv'); AN=pd.read_csv(DAT+'tab3_forecast_announced.csv')
order=['(0, 68]','(68, 72]','(72, 75]','(75, 77]','(77, 79]','(79, 81]','(81, 90]']; labs=['≤68','68–72','72–75','75–77','77–79','79–81','>81']; x=np.arange(len(order))
def get(out,lag):
    r=B[(B.outcome==out)&(B.lag==lag)].set_index('b').reindex(order); return r['mean'].values*100,r['sem'].values*100*1.96
fig,ax=plt.subplots(2,2,figsize=(7.48,4.9),gridspec_kw=dict(wspace=0.30,hspace=0.48,left=0.085,right=0.985,bottom=0.10,top=0.95))
a=ax[0,0]; a.axhline(0,color=DK,lw=0.8)
for lag,col,off,lab in (('tmax',MID,-0.18,'Same day'),('tmax_l1',RED,0.0,'1 day earlier'),('tmax_l2',BLUE,0.18,'2 days earlier')):
    m,e=get('H',lag); a.errorbar(x+off,m,yerr=e,fmt='o',ms=3.5,color=col,elinewidth=0.8,capsize=0,label=lab)
a.set_xticks(x); a.set_xticklabels(labs); a.set_xlabel('Daily maximum THI'); a.set_ylabel('Difference in herd deviation (%)'); a.set_ylim(-4.6,1.2); a.legend(loc='lower left',title='THI of',title_fontsize=7,handletextpad=0.2); panel(a,'a')
b=ax[0,1]; b.axhline(0,color=DK,lw=0.8)
m,e=get('Hraw','tmax_l1'); b.errorbar(x-0.12,m,yerr=e,fmt='o',ms=3.5,mfc='white',mec=DK,ecolor=MID,elinewidth=0.8,capsize=0,label='Daily sum of recorded milkings')
m,e=get('H','tmax_l1'); b.errorbar(x+0.12,m,yerr=e,fmt='o',ms=3.5,color=RED,elinewidth=0.8,capsize=0,label='Session-aware deviation D')
b.set_xticks(x); b.set_xticklabels(labs); b.set_xlabel('Daily maximum THI, 1 day earlier'); b.set_ylabel('Difference in herd deviation (%)'); b.set_ylim(-11.5,2.5); b.legend(loc='lower left',handletextpad=0.2); panel(b,'b')
c=ax[1,0]; PF=pd.read_csv(DAT+'tab3_per_farm_all.csv'); PF=PF[PF.tau==76].sort_values('farm',ascending=False).reset_index(drop=True); yy=np.arange(len(PF)); c.axvline(0,color=DK,lw=0.8)
c.errorbar(PF.H,yy,xerr=[PF.H-PF.H_lo,PF.H_hi-PF.H],fmt='o',ms=3.8,color=RED,elinewidth=0.8,capsize=0)
c.set_yticks(yy); c.set_yticklabels(PF.farm); c.set_xlabel('Change in herd deviation per THI unit above 76 (%)'); c.grid(axis='y',visible=False); panel(c,'c')
d=ax[1,1]; d.axhline(0,color=DK,lw=0.8); A0=AN[AN.lead==0]; A1=AN[AN.lead>0]
d.errorbar(A0.lead,A0['diff']*100,yerr=A0.se*100*1.96,fmt='o',ms=3.8,mfc='white',mec=DK,ecolor=MID,elinewidth=0.8,capsize=0,label='Reanalysis')
d.errorbar(A1.lead,A1['diff']*100,yerr=A1.se*100*1.96,fmt='o',ms=3.8,color=RED,elinewidth=0.8,capsize=0,label='Forecast')
d.set_xticks(range(0,8)); d.set_xlim(-0.6,7.6); d.set_ylim(-2.6,0.6); d.set_xlabel('Forecast lead time (days)'); d.set_ylabel('Difference in herd deviation,\nnext 2 days (%)'); d.legend(loc='upper right',handletextpad=0.2,borderaxespad=0.2,ncol=2,columnspacing=1.0); panel(d,'d')
save(fig,'Fig3')
