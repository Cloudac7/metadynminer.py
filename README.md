# metadynminer.py

Metadynminer is a package designed to help you analyse output HILLS files from PLUMED metadynamics simulations. It is based on Metadynminer package for R programming language, but it is not just a port from R to Python, as it is updated and improved in many aspects. It supports HILLS files with one, two or three collective variables. 

install:

```
pip install metadynminer
```

To support 3d plotting, please install with pyvista:

```
pip install metadynminer[3dplot]
```

> For MacOS arm64 users:
> 
> Please install vtk first using conda:
> 
> ```
> conda install vtk
> ```
>
> For there is no arm64 release for vtk now, as described [here](https://github.com/pyvista/pyvista/issues/4305).

or 
```
conda install -c jan8be metadynminer
```

Short sample code:

```python
from metadynminer.hills import Hills
from metadynminer.fes import Fes
from metadynminer.minima import Minima
from metadynminer.profile import FEProfile

# load your HILLS file
hills = Hills(name="HILLS", periodic=[True,True])

# compute the free energy surface using the fast Bias Sum Algorithm
fes = Fes(hills)

# you can also use slower (but exact) algorithm to sum the hills and compute the free energy surface 
# with the option original=True. This algorithm was checked and it gives the same result 
# (to the machine level precision) as the PLUMED sum_hills function (for plumed v2.8.0)
fes2 = Fes(hills, original=True)

# visualize the free energy surface
fes.plot()

# find local minima on the FES and print them
minima = Minima(fes)
print(minima.minima)

# You can also plot free energy profile to see, how the differences between each minima were evolving 
# during the simulation. 
fep = FEProfile(minima, hills)
fep.plot()
```

These functions can be easily customized with many parameters. You can learn more about that later in the documentation. 
There are also other predefined functions allowing you for example to remove a CV from existing FES or enhance your presentation with animated 3D FES. 
