import numpy as np
from optical_rectification import definitions, par

def test_index():
    """
    test dispersion curve
    """
    w = np.linspace(2*np.pi*1e-9, 2*np.pi*10, 2**9)  # rad/s
    try:
        n = definitions.Index(
        	w, param=par.p, s=par.s
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
