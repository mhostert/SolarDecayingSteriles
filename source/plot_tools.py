import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patches as patches
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.font_manager import *

import scipy

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

################################################
def interp_grid(x,y,z, fine_gridx=False, fine_gridy=False, logx=False, logy=False, method='interpolate', smear_stddev=False):

    # default
    if not fine_gridx:
        fine_gridx = 100
    if not fine_gridy:
        fine_gridy = 100

    # log scale x
    if logx:
        xi = np.logspace(np.min(np.log10(x)), np.max(np.log10(x)), fine_gridx)
    else: 
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)
    
    # log scale y
    if logy:
        y = -np.log(y)
        yi = np.logspace(np.min(np.log10(y)), np.max(np.log10(y)), fine_gridy)

    else:
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    
    Xi, Yi = np.meshgrid(xi, yi)
    if logy:
        Yi = np.exp(-Yi)

    # triangulation
    if method=='triangulation':
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Zi = interpolator(Xi, Yi)
    
    elif method=='interpolate':
        Zi = scipy.interpolate.griddata((x, y), z,\
                                        (xi[None,:], yi[:,None]),\
                                        method='linear', rescale =True)        
    else:
        print(f"Method {method} not implemented.")
    
    # gaussian smear -- not recommended
    if smear_stddev:
            Zi = scipy.ndimage.filters.gaussian_filter(Zi, smear_stddev, mode='nearest', order = 0, cval=0)
    
    return Xi, Yi, Zi
