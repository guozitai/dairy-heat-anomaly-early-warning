from figstyle import *
import pandas as pd, numpy as np, os
H=pd.read_csv(DAT+'fig1_interval_hist36.csv')
fig,axs=plt.subplots(2,4,figsize=(7.48,3.6),sharex=True,sharey=True,gridspec_kw=dict(wspace=0.12,hspace=0.22,left=0.075,right=0.99,bottom=0.12,top=0.97))
for i,a in enumerate(axs.flat):
    f=f'Farm {i+1}'; y=H[f]/H[f].sum()*100; a.step(H.h,y,where='mid',color=DK,lw=0.8); a.set_yscale('log'); a.set_ylim(0.003,80); a.set_xlim(0,34); a.set_xticks([0,8,16,24,32]); a.text(0.97,0.92,f,transform=a.transAxes,ha='right',va='top',fontsize=7)
    if i>=4: a.set_xlabel('Interval (h)')
    if i%4==0: a.set_ylabel('Milkings per bin (%)')
save(fig,'FigS1')
if os.path.exists(DAT+'ew_calibration.csv'):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    C=pd.read_csv(DAT+'ew_calibration.csv')
    fig,ax=plt.subplots(1,2,figsize=(6.0,2.9),gridspec_kw=dict(wspace=0.34,left=0.10,right=0.985,bottom=0.16,top=0.90))
    for a,(task,lab,lim,tk) in zip(ax,(('onset','a',(0.5,50),[1,2,5,10,20,40]),('disease','b',(0.03,12),[0.1,0.3,1,3,10]))):
        z=C[C.task==task]
        for f,g in z.groupby('fold'): a.plot(g.p*100,g.y*100,color=GREY,lw=0.6,zorder=1)
        t=z.assign(wp=z.p*z.n,wy=z.y*z.n).groupby('bin')[['wp','wy','n']].sum(); g=pd.DataFrame({'p':t.wp/t.n,'y':t.wy/t.n})
        a.plot(g.p*100,g.y*100,'o-',color=RED,ms=3.5,lw=1.0,zorder=3)
        a.plot(lim,lim,color=DK,lw=0.7,zorder=2); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(*lim); a.set_ylim(*lim)
        for axis in (a.xaxis,a.yaxis):
            axis.set_major_locator(FixedLocator(tk)); axis.set_major_formatter(FixedFormatter([str(v) for v in tk])); axis.set_minor_formatter(NullFormatter())
        a.set_xlabel('Predicted probability (%)'); a.set_ylabel('Observed frequency (%)'); panel(a,lab,x=-0.2,y=1.02)
    save(fig,'FigS2')
if os.path.exists(DAT+'tab3_dlag.csv'):
    L=pd.read_csv(DAT+'tab3_dlag.csv'); L=L[L.tau==76]
    fig,a=plt.subplots(1,1,figsize=(5.2,2.9),gridspec_kw=dict(left=0.13,right=0.985,bottom=0.17,top=0.95))
    a.axhline(0,color=DK,lw=0.8); a.axvline(0.5,color=GREY,lw=0.8,zorder=0)
    for o,col,off,lab in (('H',RED,-0.12,'Reference window t − 8 to t − 2'),('HL',BLUE,0.12,'Reference window t − 21 to t − 9')):
        z=L[L.outcome==o].sort_values('day'); a.errorbar(-z.day+off,z.coef,yerr=1.96*z.se_fm,fmt='o',ms=3.5,color=col,elinewidth=0.8,capsize=0,label=lab)
    a.set_xticks(range(-10,5)); a.set_xticklabels([('+%d'%k if k>0 else ('0' if k==0 else '−%d'%(-k))) for k in range(-10,5)]); a.set_xlim(-10.6,4.6)
    a.set_xlabel('Day of THI relative to the day of yield'); a.set_ylabel('Herd deviation per THI unit above 76 (%)'); a.set_ylim(-0.62,0.62); a.legend(loc='upper right',handletextpad=0.2,borderaxespad=0.1)
    save(fig,'FigS3')
