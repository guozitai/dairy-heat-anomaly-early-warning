from figstyle import *
import pandas as pd, numpy as np
H=pd.read_csv(DAT+'fig1_interval_hist36.csv'); Q=pd.read_csv(DAT+'fig1_gap_ratio_quantiles.csv'); T=pd.read_csv(DAT+'tab_farm_descriptives.csv').rename(columns={'drop_days':'flag_D'})
fig,ax=plt.subplots(1,3,figsize=(7.48,2.55),gridspec_kw=dict(width_ratios=[1.15,1.05,0.85],wspace=0.42,left=0.075,right=0.985,bottom=0.21,top=0.90))
a=ax[0]; tot=H['n_all_intervals'].iloc[0]; y=H['all']/tot*100
a.step(H.h,y,where='mid',color=DK,lw=0.9); a.set_yscale('log'); a.set_ylim(0.003,80); a.set_xlim(0,34); a.set_xticks([0,4,8,12,16,20,24,28,32])
a.axvspan(5,11.5,color=GREY,alpha=0.35,lw=0,zorder=0)
a.set_xlabel('Interval since previous recorded milking (h)'); a.set_ylabel('Milkings per 0.25-h bin (%)')
for x0,lab in ((8,'8 h'),(16,'16 h'),(24,'24 h')):
    yy=y[(H.h>x0-1.5)&(H.h<x0+1.5)].max(); a.text(x0,yy*1.5,lab,ha='center',va='bottom',fontsize=6.5)
panel(a,'a')
b=ax[1]; Q=Q.set_index('cat').loc[['5-11.5 h','11.5-14 h','14-18 h','18-22 h','22-26 h','26-34 h']]; x=np.arange(len(Q)); cols=[MID,DK,DK,DK,DK,DK]
for yv,lab in ((1,None),(2,'2'),(3,'3')): b.axhline(yv,color=DK if yv==1 else MID,lw=0.8 if yv==1 else 0.6,ls='-' if yv==1 else (0,(4,3)),zorder=1)
for i,(c,r) in enumerate(Q.iterrows()):
    b.plot([i,i],[r.q05,r.q95],color=cols[i],lw=0.8,zorder=2); b.plot([i,i],[r.q25,r['q75']],color=cols[i],lw=3.2,solid_capstyle='butt',zorder=3); b.plot(i,r['median'],'o',ms=3.6,mfc='white',mec=cols[i],mew=0.9,zorder=4)
b.set_xticks(x); b.set_xticklabels(['5–11.5','11.5–14','14–18','18–22','22–26','26–34'],fontsize=6.3); b.set_ylim(0.4,3.2); b.set_yticks([0.5,1,1.5,2,2.5,3]); b.set_xlabel('Interval since previous recorded milking (h)'); b.set_ylabel('Yield / usual yield of the shift'); b.grid(axis='x',visible=False)
b.text(5.45,2.03,'omitted: one',fontsize=6,ha='right',va='bottom',color=DK); b.text(5.45,3.03,'two',fontsize=6,ha='right',va='bottom',color=DK)
panel(b,'b',x=-0.2)
c=ax[2]; T=T.sort_values('farm',ascending=False).reset_index(drop=True); yy=np.arange(len(T))
c.hlines(yy,T.flag_D*100,T.flag_raw*100,color=GREY,lw=1.0,zorder=1)
c.scatter(T.flag_raw*100,yy,s=22,facecolor='white',edgecolor=DK,lw=0.8,zorder=2)
c.scatter(T.flag_D*100,yy,s=22,color=RED,zorder=3)
c.set_yticks(yy); c.set_yticklabels(T.farm); c.set_xlim(0,34); c.set_xlabel('Cow-days below 80% (%)'); c.grid(axis='y',visible=False)
c.set_ylim(-0.6,len(T)-0.2); c.text(0.8,len(T)-0.55,'Session-aware D',color=RED,fontsize=6.5,ha='left',va='bottom'); c.text(33.5,len(T)-0.55,'Daily sum',color=DK,fontsize=6.5,ha='right',va='bottom')
panel(c,'c',x=-0.30)
save(fig,'Fig1')
