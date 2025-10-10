import numpy as np
from optical_rectification import definitions, par

def test_index():
    """
    test dispersion curve
    """
    k = par.k
    n_inf = par.param[0]
    p = np.array(par.param[1:])
    w = np.linspace(2*np.pi*1e-12, 2*np.pi*20e12, 2**9)  # rad/s
    try:
        n = definitions.Index(
        	w, p[:,0], p[:,1], p[:,2], n_inf, k
        	);
    except (TypeError, ValueError, ZeroDivisionError, AttributeError) as e:
        print(f"Error running calculation: {e}");
        return None
    except Exception as e:
        print(f"Unexpected error: {e}");

    return n

n = test_index()
print(type(n),);
print(" The functions n.n(), n.alpha() have dimensions ", 
	np.shape( n.n() ), " ,",np.shape( n.alpha() )
	)