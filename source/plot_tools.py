import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D


fsize=11
rc('text', usetex=True)
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.4*3.7,1.5*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
my_hatch = 'xxxxxxxxxxxxxxxx'
rcParams.update(rcparams)
axes_form  = [0.15,0.19,0.82,0.7]

def get_std_fig(axes_form=axes_form):
	fig = plt.figure()
	ax = fig.add_axes(axes_form)
	ax.patch.set_alpha(0.0)
	ax.set_rasterization_zorder(0)
	return fig, ax

def save_figs(fig_name):
	plt.savefig(fig_name,dpi=300)
	plt.savefig(fig_name.replace("pdf","png"),dpi=300)

def to_scientific_notation(number):
	a, b = '{:.4E}'.format(number).split('E')
	b = int(b)
	a = float(a)
	return r'$%.0f \times 10^{%i}$'%(a,b)	