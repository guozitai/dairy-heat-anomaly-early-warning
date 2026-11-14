from figstyle import *
import pandas as pd, numpy as np
O=pd.read_csv(DAT+'ew_onset.csv'); Dz=pd.read_csv(DAT+'ew_disease.csv'); Lt=pd.read_csv(DAT+'ew_leadtime.csv')
key=lambda s:(int(s.split()[1]),int(s.split()[2])); folds=sorted(O.fold.unique(),key=key); x=np.arange(len(folds)); labs=[f"F{f.split()[1]}\n’{f.split()[2][2:]}" for f in folds]
fig=plt.figure(figsize=(7.48,6.9)); gs=fig.add_gridspec(3,3,height_ratios=[1,1,0.78],hspace=0.50,wspace=0.35,left=0.075,right=0.99,bottom=0.065,top=0.962)
a=fig.add_subplot(gs[0,:]); g=lambda T,c,col: T[T.config==c].set_index('fold').loc[folds,col].values*100
a.axhline(g(O,'R_deviation','chance10').mean(),color=DK,lw=0.8)
r=g(O,'R_deviation','sens10'); m=g(O,'M_ref','sens10'); a.vlines(x,r,m,color=GREY,lw=1.2,zorder=1)
a.plot(x,g(O,'M_weather+DIM+parity','sens10'),'^',ms=4,mfc='white',mec=BLUE,mew=0.9,label='Boosting: observed weather, DIM and parity',zorder=2)
a.errorbar(x,r,yerr=[r-g(O,'R_deviation','lo'),g(O,'R_deviation','hi')-r],fmt='o',ms=4.2,mfc='white',mec=DK,ecolor=DK,elinewidth=0.8,capsize=0,label='Deviation rule',zorder=3)
a.errorbar(x,m,yerr=[m-g(O,'M_ref','lo'),g(O,'M_ref','hi')-m],fmt='o',ms=4.2,color=RED,elinewidth=0.8,capsize=0,label='Boosting: main model',zorder=4)
a.set_xticks(x); a.set_xticklabels(labs); a.set_ylabel('Onsets on 10% lists (%)'); a.set_ylim(5,None); a.set_xlim(-0.6,len(folds)-0.4); a.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,handletextpad=0.3,columnspacing=1.6,borderaxespad=0.2); panel(a,'a',x=-0.07,y=1.0)
b=fig.add_subplot(gs[1,:]); r=g(Dz,'R_deviation','case_sens5'); m=g(Dz,'M_ref','case_sens5'); c0=g(Dz,'R_dailysum_cohort_ratio','case_sens5')
b.vlines(x,np.minimum(c0,np.minimum(r,m)),np.maximum(r,m),color=GREY,lw=1.2,zorder=1)
b.plot(x,g(Dz,'R_deviation','chance5'),'_',ms=9,color=MID,mew=1.2,label='Lists drawn at random',zorder=2)
b.plot(x,c0,'s',ms=4,mfc='white',mec=BLUE,mew=0.9,label='Training-cohort ratio of daily sum',zorder=2)
b.errorbar(x-0.09,r,yerr=[r-g(Dz,'R_deviation','cs5_lo'),g(Dz,'R_deviation','cs5_hi')-r],fmt='o',ms=4.2,mfc='white',mec=DK,ecolor=DK,elinewidth=0.8,capsize=0,label='Deviation rule',zorder=3)
b.errorbar(x+0.09,m,yerr=[m-g(Dz,'M_ref','cs5_lo'),g(Dz,'M_ref','cs5_hi')-m],fmt='o',ms=4.2,color=RED,elinewidth=0.8,capsize=0,label='Boosting: main model',zorder=4)
b.set_xticks(x); b.set_xticklabels(labs); b.set_ylabel('Cases detected by 5% lists (%)'); b.set_ylim(0,None); b.set_xlim(-0.6,len(folds)-0.4); b.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=4,handletextpad=0.3,columnspacing=1.6,borderaxespad=0.2); panel(b,'b',x=-0.07,y=1.0)
c=fig.add_subplot(gs[2,0:2])
lt=Lt.assign(w=Lt.on_list*Lt.n,wc=Lt.chance*Lt.n).groupby(['config','days_before'])[['w','wc','n']].sum(); lt['p']=lt.w/lt.n*100; lt['c']=lt.wc/lt.n*100
ch=lt.loc['R_deviation']; c.plot(-ch.index,ch.c,color=MID,ls=(0,(4,3)),lw=1.0,label='Lists drawn at random')
for cfg,col,lab,kw in (('R_deviation',DK,'Deviation rule',dict(mfc='white')),('M_ref',RED,'Boosting: main model',dict())):
    z=lt.loc[cfg]; c.plot(-z.index,z.p,color=col,lw=1.2,marker='o',ms=3.6,label=lab,**kw)
c.set_xticks([-7,-6,-5,-4,-3,-2,-1]); c.set_xlabel('Days before the veterinary record'); c.set_ylabel('Cows on the 5% list (%)'); c.set_ylim(0,None); c.legend(loc='upper left',handletextpad=0.3); panel(c,'c',x=-0.115,y=1.0)
save(fig,'Fig4')
