name = "metadynminer"
__version__ = "0.1.3"
__author__ = 'Jan Beránek'
"""
Metadynminer is a package designed to help you analyse output HILLS files from PLUMED metadynamics simulations. It is based on Metadynminer package for R programming language, but it is not just a port from R to Python, as it is updated and improved in many aspects. It supports HILLS files with one, two or three collective variables. 

Short sample code:

```python
# load your HILLS file
hillsfile = metadynminer.Hills(name="HILLS", periodic=[True,True])

# compute the free energy surface using the fast Bias Sum Algorithm
fes = metadynminer.Fes(hillsfile)

# you can also use slower (but exact) algorithm to sum the hills and compute the free energy surface 
# with the option original=True. This algorithm was checked and it gives the same result 
# (to the machine level precision) as the PLUMED sum_hills function (for plumed v2.8.0)
fes2 = metadynminer.Fes(hillsfile, original=True)

# visualize the free energy surface
fes.plot()

# find local minima on the FES and print them
minima = metadynminer.Minima(fes)
print(minima.minima)

# You can also plot free energy profile to see, how the differences between each minima were evolving 
# during the simulation. 
fep = metadynminer.FEProfile(minima, hillsfile)
fep.plot()
```

These functions can be easily customized with many parameters. You can learn more about that later in the documentation. 
There are also other predefined functions allowing you for example to remove a CV from existing FES or enhance your presentation with animated 3D FES. 
"""
import sys

try:
    import numpy as np
except:
    print("Error while loading numpy")
    sys.exit(1)
try:
    from matplotlib import pyplot as plt
except:
    print("Error while loading matplotlib pyplot")
    sys.exit(1)
try:
    from matplotlib import colormaps as cm
except:
    print("Error while loading matplotlib colormaps")
    sys.exit(1)
try:
    import pandas as pd
except:
    print("Error while loading pandas")
    sys.exit(1)
