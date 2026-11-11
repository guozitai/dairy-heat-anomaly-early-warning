import matplotlib, glob
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in glob.glob('/usr/share/fonts/truetype/liberation/Liberation*.ttf'): fm.fontManager.addfont(_f)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7.5,'axes.linewidth':0.6,'axes.edgecolor':'black','axes.spines.top':False,'axes.spines.right':False,
 'xtick.major.size':0,'ytick.major.size':0,'xtick.labelsize':7,'ytick.labelsize':7,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
 'axes.labelsize':8,'legend.fontsize':7,'legend.frameon':False,'svg.fonttype':'none','axes.grid':True,'grid.color':'0.92','grid.linewidth':0.5,'axes.axisbelow':True,'lines.linewidth':1.0})
RED='#C8464A'; BLUE='#416A99'; GREY='#D0D0D0'; DK='#4D4D4D'; MID='#8C8C8C'
OUT='/work/cea_sat/rescue/ms/figs/'; DAT='/work/cea_sat/rescue/v2/out/'
def panel(ax,letter,x=-0.16,y=1.04): ax.text(x,y,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='left')
def save(fig,name):
    import subprocess
    fig.savefig(OUT+name+'.svg'); fig.savefig(OUT+name+'.pdf')
    subprocess.run(['rsvg-convert','-d','300','-p','300','-b','white','-o',OUT+name+'.png',OUT+name+'.svg'],check=True)
    plt.close(fig)
