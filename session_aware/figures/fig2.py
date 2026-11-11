from figstyle import *
import pandas as pd, numpy as np
T=pd.read_csv(DAT+'fig2_trajectories.csv'); VT=pd.read_csv(DAT+'tab2_validity.csv').set_index('group')
groups=[('All diseases',RED),('Udder',RED),('Digestive',RED),('Abomasal and metabolic',RED),('Locomotor',RED),('Respiratory',RED),('Reproductive tract',RED),('Oestrus recorded',BLUE),('Insemination',BLUE),('Vaccination',BLUE),('Hoof trimming',BLUE),('Pen change',BLUE)]
LAB={'All diseases':'All diseases','Udder':'Udder disorders','Digestive':'Digestive disorders','Abomasal and metabolic':'Abomasal and\nmetabolic disorders','Locomotor':'Locomotor\ndisorders','Respiratory':'Respiratory disease','Reproductive tract':'Reproductive tract','Oestrus recorded':'Oestrus observation','Insemination':'Insemination','Vaccination':'Vaccination','Hoof trimming':'Hoof trimming','Pen change':'Pen change'}
fig,axs=plt.subplots(3,4,figsize=(7.48,5.2),sharex=True,gridspec_kw=dict(wspace=0.32,hspace=0.26,left=0.075,right=0.99,bottom=0.085,top=0.975))
for i,(g,col) in enumerate(groups):
    a=axs.flat[i]; c=T[(T.group==g)&(T.arm=='case')&(T.k<=3)].sort_values('k'); k=T[(T.group==g)&(T.arm=='control')&(T.k<=3)].sort_values('k')
    n1=c.loc[c.k==-1,'size'].iloc[0]; keep=c[c['size']>=0.5*n1].k; c=c[c.k.isin(keep)]; k=k[k.k.isin(keep)]   # days on which at least half of the cows had a day deviation
    a.axhline(0,color=DK,lw=0.6); a.axvline(0,color=GREY,lw=0.8,zorder=0)
    a.plot(k.k,(np.exp(k['mean'])-1)*100,color=MID,lw=1.0,marker='o',ms=2.2,mfc='white',mew=0.6)
    a.fill_between(c.k,(np.exp(c['mean']-1.96*c['sem'])-1)*100,(np.exp(c['mean']+1.96*c['sem'])-1)*100,color=col,alpha=0.18,lw=0)
    pc=(np.exp(c['mean'])-1)*100; a.plot(c.k,pc,color=col,lw=1.1,marker='o',ms=2.6)
    lo=min(-8,np.floor((pc.min()-4)/10)*10); a.set_ylim(lo,12 if pc.max()<10 else np.ceil((pc.max()+4)/10)*10)
    n=int(VT.loc[g,'pre3_n'])
    a.text(0.03,0.06,f"{LAB[g]}\nn = {n:,}",transform=a.transAxes,fontsize=6.8,va='bottom',ha='left')
    a.set_xticks([-9,-6,-3,0,3]); a.set_xlim(-10.5,3.5)
    if i%4==0: a.set_ylabel('Deviation (%)')
    if i>=8: a.set_xlabel('Days from record')
save(fig,'Fig2')
