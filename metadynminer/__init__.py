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
try:
    import numpy as np
except:
    print("Error while loading numpy")
    exit()
try:
    from matplotlib import pyplot as plt
except:
    print("Error while loading matplotlib pyplot")
    exit()
try:
    from matplotlib import colormaps as cm
except:
    print("Error while loading matplotlib colormaps")
    exit()
try:
    import pandas as pd
except:
    print("Error while loading pandas")
    exit()
try:
    import pyvista as pv
except:
    print("Error while loading pyvista")
    exit()

  
class Minima():
    """
    Object of Minima class is created to find local free energy minima on FES. 
    The FES is first divided to some number of bins, 
    (the number of bins can be set with option nbins, default is 8)
    and the absolute minima is found for each bin. Then the algorithm checks 
    if this point is really a local minimum by comparing to the surrounding points of FES.
    
    The list of minima is stored as pandas dataframe. 
    
    Command:
    ```python
    minima = metadynminer.Minima(fes=f, nbins=8)
    ```
    
    List of minima can be later called like this:
    
    ```python
    print(minima.minima)
    ```
    
    Parameters:
    
    * fes = Fes object to find the minima on
    
    * nbins (default = 8) = number of bins to divide the FES
    """
    def __init__(self, fes, nbins = 8):
        self.fes = fes.fes
        self.periodic = fes.periodic
        self.cvs = fes.cvs
        self.res = fes.res

        if self.cvs >= 1:
            self.cv1_name = fes.cv1_name
            self.cv1min = fes.cv1min
            self.cv1max = fes.cv1max
            self.cv1per = fes.cv1per
        if self.cvs >= 2:
            self.cv2min = fes.cv2min
            self.cv2max = fes.cv2max
            self.cv2_name = fes.cv2_name
            self.cv2per = fes.cv2per
        if self.cvs == 3:
            self.cv3min = fes.cv3min
            self.cv3max = fes.cv3max
            self.cv3_name = fes.cv3_name
            self.cv3per = fes.cv3per
        
        self.findminima(nbins=nbins)

    def findminima(self, nbins=8):
        """
        Internal method for finding local minima on FES.
        """
        if int(nbins) != nbins:
            nbins = int(nbins)
            print(f"Number of bins must be an integer, it will be set to {nbins}.")
        if self.res%nbins != 0:
            print("Error: Resolution of FES must be divisible by number of bins.")
            return None
        if nbins > self.res/2:
            print("Error: Number of bins is too high.")
            return None
        bin_size = int(self.res/nbins)

        if self.cvs >= 1:
            if not self.periodic[0]:
                cv1min = self.cv1min - (self.cv1max-self.cv1min)*0.15
                cv1max = self.cv1max + (self.cv1max-self.cv1min)*0.15
            else:
                cv1min = self.cv1min
                cv1max = self.cv1max 
        if self.cvs >=2:
            if not self.periodic[1]:
                cv2min = self.cv2min - (self.cv2max-self.cv2min)*0.15
                cv2max = self.cv2max + (self.cv2max-self.cv2min)*0.15
            else:
                cv2min = self.cv2min
                cv2max = self.cv2max 
        if self.cvs == 3:
            if not self.periodic[2]:
                cv3min = self.cv3min - (self.cv3max-self.cv3min)*0.15
                cv3max = self.cv3max + (self.cv3max-self.cv3min)*0.15
            else:
                cv3min = self.cv3min
                cv3max = self.cv3max

        self.minima = []
        if self.cvs == 1:
            for bin1 in range(0,nbins):
                
                fes_slice = self.fes[bin1*bin_size:(bin1+1)*bin_size]
                bin_min = np.min(fes_slice)
                argmin = np.argmin(fes_slice)
                # indexes of global minimum of a bin
                bin_min_arg_cv1 = int(argmin%bin_size)
                # indexes of that minima in the original fes (indexes +1)
                min_cv1_b = int(bin_min_arg_cv1+bin1*bin_size)
                if (bin_min_arg_cv1 > 0 and bin_min_arg_cv1<(bin_size-1)):
                    min_cv1 = (((min_cv1_b+0.5)/self.res)*(cv1max-cv1min))+cv1min
                    if self.minima == []:
                        self.minima=np.array([round(bin_min, 6), int(min_cv1_b), round(min_cv1, 6)])
                    else:
                        self.minima=np.vstack((self.minima, np.array([round(bin_min, 6), int(min_cv1_b), round(min_cv1, 6)])))
                else:
                    around = []
                    min_cv1_b_low = min_cv1_b - 1
                    if min_cv1_b_low == -1:
                        if self.periodic[0]:
                            min_cv1_b_low = self.res - 1
                        else:
                            min_cv1_b_low = float("nan")

                    min_cv1_b_high = min_cv1_b + 1
                    if min_cv1_b_high == self.res:
                        if self.periodic[0]:
                            min_cv1_b_high = 0
                        else:
                            min_cv1_b_high = float("nan")

                    #1_b_low
                    if not(np.isnan(min_cv1_b_low)):
                        around.append(self.fes[min_cv1_b_low])
                    #1_b_high
                    if not(np.isnan(min_cv1_b_high)):
                        around.append(self.fes[min_cv1_b_high])
                    
                    if bin_min < np.min(around):
                        min_cv1 = (((min_cv1_b+0.5)/self.res)*(cv1max-cv1min))+cv1min
                        if self.minima == []:
                            self.minima=np.array([round(bin_min, 6), int(min_cv1_b), round(min_cv1, 6)])
                        else:
                            self.minima=np.vstack((self.minima, np.array([round(bin_min, 6), int(min_cv1_b), round(min_cv1, 6)])))
            
        elif self.cvs == 2:
            for bin1 in range(0,nbins):
                for bin2 in range(0,nbins):
                    fes_slice = self.fes[bin1*bin_size:(bin1+1)*bin_size,
                                         bin2*bin_size:(bin2+1)*bin_size]
                    bin_min = np.min(fes_slice)
                    argmin = np.argmin(fes_slice)
                    # indexes of global minimum of a bin
                    bin_min_arg = np.unravel_index(np.argmin(fes_slice), fes_slice.shape)
                    # indexes of that minima in the original fes (indexes +1)
                    min_cv1_b = int(bin_min_arg[0]+bin1*bin_size)
                    min_cv2_b = int(bin_min_arg[1]+bin2*bin_size)
                    if (bin_min_arg[0] > 0 and bin_min_arg[0]<(bin_size-1)) \
                                    and (bin_min_arg[1] > 0 and bin_min_arg[1]<(bin_size-1)):
                        min_cv1 = (((min_cv1_b+0.5)/self.res)*(cv1max-cv1min))+cv1min
                        min_cv2 = (((min_cv2_b+0.5)/self.res)*(cv2max-cv2min))+cv2min
                        if self.minima == []:
                            self.minima=np.array([round(bin_min, 6), int(min_cv1_b),\
                                                  int(min_cv2_b), round(min_cv1, 6), round(min_cv2, 6)])
                        else:
                            self.minima=np.vstack((self.minima, np.array([round(bin_min, 6), int(min_cv1_b), \
                                                                          int(min_cv2_b), round(min_cv1, 6), round(min_cv2, 6)])))
                    else:
                        around = []
                        min_cv1_b_low = min_cv1_b - 1
                        if min_cv1_b_low == -1:
                            if self.periodic[0]:
                                min_cv1_b_low = self.res - 1
                            else:
                                min_cv1_b_low = float("nan")

                        min_cv1_b_high = min_cv1_b + 1
                        if min_cv1_b_high == self.res:
                            if self.periodic[0]:
                                min_cv1_b_high = 0
                            else:
                                min_cv1_b_high = float("nan")

                        min_cv2_b_low = min_cv2_b - 1
                        if min_cv2_b_low == -1:
                            if self.periodic[0]:
                                min_cv2_b_low = self.res - 1
                            else:
                                min_cv2_b_low = float("nan")

                        min_cv2_b_high = min_cv2_b + 1
                        if min_cv2_b_high == self.res:
                            if self.periodic[0]:
                                min_cv2_b_high = 0
                            else:
                                min_cv2_b_high = float("nan")
                        #1_b_low
                        if not(np.isnan(min_cv1_b_low)):
                            if not(np.isnan(min_cv2_b_low)):
                                around.append(self.fes[min_cv1_b_low, min_cv2_b_low])
                            around.append(self.fes[min_cv1_b_low,min_cv2_b])
                            if not(np.isnan(min_cv2_b_high)):
                                around.append(self.fes[min_cv1_b_low, min_cv2_b_high])
                        #1_b
                        if not(np.isnan(min_cv2_b_low)):
                            around.append(self.fes[min_cv1_b, min_cv2_b_low])
                        if not(np.isnan(min_cv2_b_high)):
                            around.append(self.fes[min_cv1_b, min_cv2_b_high])
                        #1_b_high
                        if not(np.isnan(min_cv1_b_high)):
                            if not(np.isnan(min_cv2_b_low)):
                                around.append(self.fes[min_cv1_b_high, min_cv2_b_low])
                            around.append(self.fes[min_cv1_b_high, min_cv2_b])
                            if not(np.isnan(min_cv2_b_high)):
                                around.append(self.fes[min_cv1_b_high, min_cv2_b_high])
                        if bin_min < np.min(around):
                            min_cv1 = (((min_cv1_b+0.5)/self.res)*(cv1max-cv1min))+cv1min
                            min_cv2 = (((min_cv2_b+0.5)/self.res)*(cv2max-cv2min))+cv2min
                            if self.minima == []:
                                self.minima=np.array([round(bin_min, 6), int(min_cv1_b), int(min_cv2_b), \
                                                      round(min_cv1, 6), round(min_cv2, 6)])
                            else:
                                self.minima=np.vstack((self.minima, np.array([round(bin_min, 6), int(min_cv1_b), \
                                                                              int(min_cv2_b), round(min_cv1, 6), round(min_cv2, 6)])))
        elif self.cvs == 3:
            for bin1 in range(0,nbins):
                for bin2 in range(0,nbins):
                    for bin3 in range(0, nbins):
                        fes_slice = self.fes[bin1*bin_size:(bin1+1)*bin_size,
                                             bin2*bin_size:(bin2+1)*bin_size, 
                                             bin3*bin_size:(bin3+1)*bin_size]
                        bin_min = np.min(fes_slice)
                        argmin = np.argmin(fes_slice)
                        # indexes of global minimum of a bin
                        bin_min_arg = np.unravel_index(np.argmin(fes_slice), fes_slice.shape)
                        # indexes of that minima in the original fes (indexes +1)
                        min_cv1_b = int(bin_min_arg[0]+bin1*bin_size)
                        min_cv2_b = int(bin_min_arg[1]+bin2*bin_size)
                        min_cv3_b = int(bin_min_arg[2]+bin3*bin_size)
                        if (bin_min_arg[0] > 0 and bin_min_arg[0]<(bin_size-1)) \
                                        and (bin_min_arg[1] > 0 and bin_min_arg[1]<(bin_size-1))\
                                        and (bin_min_arg[2] > 0 and bin_min_arg[2]<(bin_size-1)):
                            min_cv1 = (((min_cv1_b+0.5)/self.res)*(cv1max-cv1min))+cv1min
                            min_cv2 = (((min_cv2_b+0.5)/self.res)*(cv2max-cv2min))+cv2min
                            min_cv3 = (((min_cv3_b+0.5)/self.res)*(cv3max-cv3min))+cv3min
                            if self.minima == []:
                                self.minima=np.array([round(bin_min, 6), int(min_cv1_b),\
                                                      int(min_cv2_b), int(min_cv3_b), round(min_cv1, 6), \
                                                      round(min_cv2, 6), round(min_cv3, 6)])
                            else:
                                self.minima=np.vstack((self.minima, np.array([round(bin_min, 6), int(min_cv1_b),\
                                                      int(min_cv2_b), int(min_cv3_b), round(min_cv1, 6), \
                                                      round(min_cv2, 6), round(min_cv3, 6)])))
                        else:
                            around = []
                            min_cv1_b_low = min_cv1_b - 1
                            if min_cv1_b_low == -1:
                                if self.periodic[0]:
                                    min_cv1_b_low = self.res - 1
                                else:
                                    min_cv1_b_low = float("nan")

                            min_cv1_b_high = min_cv1_b + 1
                            if min_cv1_b_high == self.res:
                                if self.periodic[0]:
                                    min_cv1_b_high = 0
                                else:
                                    min_cv1_b_high = float("nan")

                            min_cv2_b_low = min_cv2_b - 1
                            if min_cv2_b_low == -1:
                                if self.periodic[0]:
                                    min_cv2_b_low = self.res - 1
                                else:
                                    min_cv2_b_low = float("nan")

                            min_cv2_b_high = min_cv2_b + 1
                            if min_cv2_b_high == self.res:
                                if self.periodic[0]:
                                    min_cv2_b_high = 0
                                else:
                                    min_cv2_b_high = float("nan")
                                                       
                            min_cv3_b_low = min_cv3_b - 1
                            if min_cv3_b_low == -1:
                                if self.periodic[2]:
                                    min_cv3_b_low = self.res - 1
                                else:
                                    min_cv3_b_low = float("nan")

                            min_cv3_b_high = min_cv3_b + 1
                            if min_cv3_b_high == self.res:
                                if self.periodic[2]:
                                    min_cv3_b_high = 0
                                else:
                                    min_cv3_b_high = float("nan")

#cv3_b
                            #1_b_low
                            if not(np.isnan(min_cv1_b_low)):
                                if not(np.isnan(min_cv2_b_low)):
                                    around.append(self.fes[min_cv1_b_low,min_cv2_b_low,min_cv3_b])
                                around.append(self.fes[min_cv1_b_low,min_cv2_b,min_cv3_b])
                                if not(np.isnan(min_cv2_b_high)):
                                    around.append(self.fes[min_cv1_b_low,min_cv2_b_high,min_cv3_b])
                            #1_b
                            if not(np.isnan(min_cv2_b_low)):
                                around.append(self.fes[min_cv1_b,min_cv2_b_low,min_cv3_b])
                            if not(np.isnan(min_cv2_b_high)):
                                around.append(self.fes[min_cv1_b,min_cv2_b_high,min_cv3_b])
                            #1_b_high
                            if not(np.isnan(min_cv1_b_high)):
                                if not(np.isnan(min_cv2_b_low)):
                                    around.append(self.fes[min_cv1_b_high,min_cv2_b_low,min_cv3_b])
                                around.append(self.fes[min_cv1_b_high,min_cv2_b,min_cv3_b])
                                if not(np.isnan(min_cv2_b_high)):
                                    around.append(self.fes[min_cv1_b_high,min_cv2_b_high,min_cv3_b])
                           
                            if not(np.isnan(min_cv3_b_low)):
                            #1_b_low
                                if not(np.isnan(min_cv1_b_low)):
                                    if not(np.isnan(min_cv2_b_low)):
                                        around.append(self.fes[min_cv1_b_low,min_cv2_b_low,min_cv3_b_low])
                                    around.append(self.fes[min_cv1_b_low,min_cv2_b,min_cv3_b_low])
                                    if not(np.isnan(min_cv2_b_high)):
                                        around.append(self.fes[min_cv1_b_low,min_cv2_b_high,min_cv3_b_low])
                                #1_b
                                if not(np.isnan(min_cv2_b_low)):
                                    around.append(self.fes[min_cv1_b,min_cv2_b_low,min_cv3_b_low])
                                if not(np.isnan(min_cv2_b_high)):
                                    around.append(self.fes[min_cv1_b,min_cv2_b_high,min_cv3_b_low])
                                #1_b_high
                                if not(np.isnan(min_cv1_b_high)):
                                    if not(np.isnan(min_cv2_b_low)):
                                        around.append(self.fes[min_cv1_b_high,min_cv2_b_low,min_cv3_b_low])
                                    around.append(self.fes[min_cv1_b_high,min_cv2_b,min_cv3_b_low])
                                    if not(np.isnan(min_cv2_b_high)):
                                        around.append(self.fes[min_cv1_b_high,min_cv2_b_high,min_cv3_b_low])
                            
                            if not(np.isnan(min_cv2_b_high)):
                                #1_b_low
                                if not(np.isnan(min_cv1_b_low)):
                                    if not(np.isnan(min_cv2_b_low)):
                                        around.append(self.fes[min_cv1_b_low,min_cv2_b_low,min_cv3_b_high])
                                    around.append(self.fes[min_cv1_b_low,min_cv2_b,min_cv3_b_high])
                                    if not(np.isnan(min_cv2_b_high)):
                                        around.append(self.fes[min_cv1_b_low,min_cv2_b_high,min_cv3_b_high])
                                #1_b
                                if not(np.isnan(min_cv2_b_low)):
                                    around.append(self.fes[min_cv1_b,min_cv2_b_low,min_cv3_b_high])
                                if not(np.isnan(min_cv2_b_high)):
                                    around.append(self.fes[min_cv1_b,min_cv2_b_high,min_cv3_b_high])
                                #1_b_high
                                if not(np.isnan(min_cv1_b_high)):
                                    if not(np.isnan(min_cv2_b_low)):
                                        around.append(self.fes[min_cv1_b_high,min_cv2_b_low,min_cv3_b_high])
                                    around.append(self.fes[min_cv1_b_high,min_cv2_b,min_cv3_b_high])
                                    if not(np.isnan(min_cv2_b_high)):
                                        around.append(self.fes[min_cv1_b_high,min_cv2_b_high,min_cv3_b_high])
                            
                            if bin_min < np.min(around):
                                min_cv1 = (((min_cv1_b+0.5)/self.res)*(cv1max-cv1min))+cv1min
                                min_cv2 = (((min_cv2_b+0.5)/self.res)*(cv2max-cv2min))+cv2min
                                min_cv3 = (((min_cv3_b+0.5)/self.res)*(cv3max-cv3min))+cv3min
                                if self.minima == []:
                                    self.minima=np.array([round(bin_min, 6), int(min_cv1_b),\
                                                      int(min_cv2_b), int(min_cv3_b), round(min_cv1, 6), \
                                                      round(min_cv2, 6), round(min_cv3, 6)])
                                else:
                                    self.minima=np.vstack((self.minima, np.array([round(bin_min, 6), int(min_cv1_b),\
                                                      int(min_cv2_b), int(min_cv3_b), round(min_cv1, 6), \
                                                      round(min_cv2, 6), round(min_cv3, 6)])))
        
        else:
            print("Fes object has unsupported number of CVs.")
        
        if len(self.minima.shape)>1:
            self.minima = self.minima[self.minima[:, 0].argsort()]

        letters = list(map(chr, range(65, 91)))
        for letter1 in range(65, 91):
            for letter2 in range(65, 91):
                letters.append(f"{chr(letter1)}{chr(letter2)}")
        if len(self.minima.shape)>1:
            if self.minima.shape[1] < len(letters):
                self.minima = np.column_stack((letters[0:self.minima.shape[0]],self.minima))
            else:
                print("Error: Too many minima to assign letters.")
        elif len(self.minima.shape) == 1:
            self.minima = np.append("A", self.minima)
        
        if self.cvs == 1:
            self.minima = pd.DataFrame(self.minima, columns = ["Minimum", "free energy", "CV1bin", "CV1 - "+self.cv1_name])
        elif self.cvs == 2:
            if len(self.minima.shape)>1:
                self.minima = pd.DataFrame(np.array(self.minima), columns = ["Minimum", "free energy", "CV1bin", "CV2bin", 
                                                               "CV1 - "+self.cv1_name, "CV2 - "+self.cv2_name])
            elif len(self.minima.shape) == 1:
                self.minima = pd.DataFrame([self.minima], columns = ["Minimum", "free energy", "CV1bin", "CV2bin", 
                                                               "CV1 - "+self.cv1_name, "CV2 - "+self.cv2_name])
        elif self.cvs == 3:
            if len(self.minima.shape)>1:
                self.minima = pd.DataFrame(np.array(self.minima), columns = ["Minimum", "free energy", "CV1bin", "CV2bin", "CV3bin", 
                                                               "CV1 - "+self.cv1_name, "CV2 - "+self.cv2_name,  "CV3 - "+self.cv3_name])
            elif len(self.minima.shape) == 1:
                self.minima = pd.DataFrame([self.minima], columns = ["Minimum", "free energy", "CV1bin", "CV2bin", "CV3bin", 
                                                               "CV1 - "+self.cv1_name, "CV2 - "+self.cv2_name,  "CV3 - "+self.cv3_name])
        

    def plot(self, png_name=None, contours=True, contours_spacing=0.0, aspect = 1.0, cmap = "jet", 
                 energy_unit="kJ/mol", xlabel=None, ylabel=None, zlabel=None, label_size=12, image_size=[10,7], 
                 color=None, vmin = 0, vmax = None, opacity=0.2, levels=None, show_points=True, point_size=4.0):
        """
        The same function as for visualizing Fes objects, but this time 
        with the positions of local minima shown as letters on the graph.
        
        ```python
        minima.plot()
        ```
        
        Parameters:
        
        * png_name = String. If this parameter is supplied, the picture of FES will be saved under this name to the current working directory.
        
        * contours (default=True) = whether contours should be shown on 2D FES
        
        * contours_spacing (default=0.0) = when a positive number is set, it will be used as spacing for contours on 2D FES. 
                Otherwise, if contours=True, there will be five equally spaced contour levels.
        
        * aspect (default = 1.0) = aspect ratio of the graph. Works with 1D and 2D FES. 
        
        * cmap (default = "jet") = Matplotlib colormap used to color 2D or 3D FES
        
        * energy_unit (default="kJ/mol") = String, used in description of colorbar
        
        * xlabel, ylabel, zlabel = Strings, if provided, they will be used as labels for the graphs
        
        * labelsize (default = 12) = size of text in labels
        
        * image_size (default = [10,7]) = List of the width and height of the picture
        
        * color = string = name of color in matplotlib, if set, the color will be used for the letters. 
                If not set, the color should be automatically either black or white, 
                depending on what will be better visible on given place on FES with given colormap (for 2D FES).
        
        * vmin (default=0) = real number, lower bound for the colormap on 2D FES
        
        * vmax = real number, upper bound for the colormap on 2D FES
        
        * opacity (default=0.2) = number between 0 and 1, is the opacity of isosurfaces of 3D FES
        
        * levels = Here you can specify list of free energy values for isosurfaces on 3D FES. 
                If not provided, default values from contours parameters will be used instead. 
        
        * show_points (default=True) = boolean, tells if points should be visualized too, instead of just the letters. Only on 3D FES. 
        
        * point_size (default=4.0) = float, sets the size of points if show_points=True
        """
        
        if vmax == None:
            vmax = np.max(self.fes)+0.01 # if the addition is smaller than 0.01, the 3d plot stops working. 
            
        if contours_spacing == 0.0:
            contours_spacing = (vmax-vmin)/5.0
        
        cmap = cm.get_cmap(cmap)
        
        cmap.set_over("white")
        cmap.set_under("white")
        
        color_set = True
        if color == None:
            color_set = False
        
        if self.cvs >= 1:
            if not self.periodic[0]:
                cv1min = self.cv1min - (self.cv1max-self.cv1min)*0.15
                cv1max = self.cv1max + (self.cv1max-self.cv1min)*0.15
            else:
                cv1min = self.cv1per[0]
                cv1max = self.cv1per[1] 
        if self.cvs >=2:
            if not self.periodic[1]:
                cv2min = self.cv2min - (self.cv2max-self.cv2min)*0.15
                cv2max = self.cv2max + (self.cv2max-self.cv2min)*0.15
            else:
                cv2min = self.cv2per[0]
                cv2max = self.cv2per[1] 
        if self.cvs == 3:
            if not self.periodic[2]:
                cv3min = self.cv3min - (self.cv3max-self.cv3min)*0.15
                cv3max = self.cv3max + (self.cv3max-self.cv3min)*0.15
            else:
                cv3min = self.cv3per[0]
                cv3max = self.cv3per[1] 
        
        if self.cvs == 1:
            plt.figure(figsize=(image_size[0],image_size[1]))
            X = np.linspace(cv1min, cv1max, self.res)
            plt.plot(X, self.fes)
            
            if not color_set:
                color = "black"
            
            ferange = np.max(self.fes) - np.min(self.fes)
            
            
            if self.minima.shape[0] == 1:
                plt.text(float(self.minima.iloc[0,3]), float(self.minima.iloc[0,1])+ferange*0.05, self.minima.iloc[0,0],
                             fontsize=label_size, horizontalalignment='center',
                             verticalalignment='bottom', c=color)
            elif self.minima.shape[0] > 1:
                for m in range(len(self.minima.iloc[:,0])):
                    plt.text(float(self.minima.iloc[m,3]), float(self.minima.iloc[m,1])+ferange*0.05, self.minima.iloc[m,0],
                             fontsize=label_size, horizontalalignment='center',
                             verticalalignment='bottom', c=color)
            
            if xlabel == None:
                plt.xlabel(f'CV1 - {self.cv1_name}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(f'free energy ({energy_unit})', size=label_size)
            else:
                plt.ylabel(ylabel, size=label_size)
            
        elif self.cvs == 2:
            fig = plt.figure(figsize=(image_size[0],image_size[1]))
            plt.imshow(np.rot90(self.fes, axes=(0,1)), cmap=cmap, interpolation='nearest', 
                       extent=[cv1min, cv1max, cv2min, cv2max], 
                       aspect = (((cv1max-cv1min)/(cv2max-cv2min))/(aspect)),
                       vmin = vmin, vmax = vmax)
            cbar = plt.colorbar()
            cbar.set_label(energy_unit, size=label_size)

            if self.minima.shape[0] == 1:
                background = cmap((float(self.minima.iloc[1])-vmin)/(vmax-vmin))
                luma = background[0]*0.2126+background[1]*0.7152+background[3]*0.0722
                if luma > 0.6 and not color_set:
                    color = "black"
                elif luma <= 0.6 and not color_set:
                    color="white"
                plt.text(float(self.minima.iloc[0,4]), float(self.minima.iloc[0,5]), self.minima.iloc[0,0],
                             fontsize=label_size, horizontalalignment='center',
                             verticalalignment='center', c=color)
            elif self.minima.shape[0] > 1:
                for m in range(len(self.minima.iloc[:,0])):
                    background = cmap((float(self.minima.iloc[m,1])-vmin)/(vmax-vmin))
                    luma = background[0]*0.2126+background[1]*0.7152+background[3]*0.0722
                    if luma > 0.6 and not color_set:
                        color = "black"
                    elif luma <= 0.6 and not color_set:
                        color="white"
                    plt.text(float(self.minima.iloc[m,4]), float(self.minima.iloc[m,5]), self.minima.iloc[m,0],
                             fontsize=label_size, horizontalalignment='center',
                             verticalalignment='center', c=color)

            if contours:
                cont = plt.contour(np.rot90(self.fes, axes=(0,1)), 
                         levels = np.arange(0, (vmax + 0.01), contours_spacing), 
                         extent=[cv1min, cv1max, cv2max, cv2min], 
                         colors = "k")
                plt.clabel(cont, levels = np.arange(0, (vmax + 0.01), contours_spacing))
            if xlabel == None:
                plt.xlabel(f'CV1 - {self.cv1_name}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(f'CV2 - {self.cv2_name}', size=label_size)
            else:
                plt.ylabel(ylabel, size=label_size)
        
        elif self.cvs == 3:
            if xlabel == None:
                xlabel = "CV1 - " + self.cv1_name
            if ylabel == None:
                ylabel = "CV2 - " + self.cv2_name
            if zlabel == None:
                zlabel = "CV3 - " + self.cv3_name
            
            min_ar = self.minima.iloc[:,5:8].values
            min_ar = min_ar.astype(np.float32)
            min_pv = pv.PolyData(min_ar)
            grid = pv.UniformGrid(
                dimensions=(self.res, self.res, self.res),
                spacing=((cv1max-cv1min)/self.res,(cv2max-cv2min)/self.res,(cv3max-cv3min)/self.res),
                origin=(cv1min, cv2min, cv3min)
            )
            grid["vol"] = self.fes.ravel(order="F")
            if levels == None:
                contours = grid.contour(np.arange(0, (vmax - 0.1), contours_spacing))
            else:
                contours = grid.contour(levels)
            fescolors = []
            for i in range(contours.points.shape[0]):
                fescolors.append(self.fes[int((contours.points[i,0]-cv1min)*self.res/(cv1max-cv1min)),
                                          int((contours.points[i,1]-cv2min)*self.res/(cv2max-cv2min)),
                                          int((contours.points[i,2]-cv3min)*self.res/(cv3max-cv3min))])
            #%% Visualization
            pv.set_plot_theme('document')
            p = pv.Plotter()
            p.add_mesh(contours, scalars=fescolors, opacity=opacity, cmap=cmap, show_scalar_bar=False, interpolate_before_map=True)
            p.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            p.add_point_labels(min_pv, self.minima.iloc[:,0], 
                   show_points=show_points, always_visible = True, 
                   point_color="black", point_size=point_size, 
                   font_size=label_size, shape=None)
            p.show()
            
        if png_name != None:
            plt.savefig(png_name)

    def make_gif(self, gif_name="FES.gif", cmap = "jet", 
                 xlabel=None, ylabel=None, zlabel=None, label_size=12, image_size=[10,7], 
                  opacity=0.2, levels=None, show_points=True, point_size=4.0, frames=64):
        """
        Equvivalent to Fes.make_gif()
        """
        if self.cvs == 3:
            values = np.linspace(np.min(self.fes)+1, np.max(self.fes), num=frames)
            grid = pv.UniformGrid(
                dimensions=(self.res, self.res, self.res),
                spacing=((self.cv1max-self.cv1min)/self.res,(self.cv2max-self.cv2min)/self.res,(self.cv3max-self.cv3min)/self.res),
                origin=(self.cv1min, self.cv2min, self.cv3min),
            )
            grid["vol"] = self.fes.ravel(order="F")
            surface = grid.contour(values[:1])
            surfaces = [grid.contour([v]) for v in values]
            surface = surfaces[0].copy()
            
            pv.set_plot_theme('document')
            plotter = pv.Plotter(off_screen=True)
            # Open a movie file
            plotter.open_gif(gif_name)

            # Add initial mesh
            plotter.add_mesh(
                surface,
                opacity=0.3,
                clim=grid.get_data_range(),
                show_scalar_bar=False,
                cmap="jet"
            )
            plotter.add_mesh(grid.outline_corners(), color="k")
            if xlabel == None and ylabel == None and zlabel == None:
                plotter.show_grid(xlabel=f"CV1 - {self.cv1_name}", ylabel=f"CV2 - {self.cv2_name}", zlabel=f"CV3 - {self.cv3_name}")
            else:
                plotter.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            if show_points:
                min_ar = self.minima.iloc[:,5:8].values
                min_ar = min_ar.astype(np.float32)
                min_pv = pv.PolyData(min_ar)
                plotter.add_point_labels(min_pv, self.minima.iloc[:,0], 
                               show_points=True, always_visible = True, 
                               pickable = True, point_color="black", 
                               point_size=4, font_size=16, shape=None)
            plotter.set_background('white')
            plotter.show(auto_close=False)

            # Run through each frame
            for surf in surfaces:
                surface.copy_from(surf)
                plotter.write_frame()  # Write this frame
            # Run through backwards
            for surf in surfaces[::-1]:
                surface.copy_from(surf)
                plotter.write_frame()  # Write this frame

            # Be sure to close the plotter when finished
            plotter.close()
        else:
            print("Error: gif_plot is only available for FES with 3 CVs.")
        
class FEProfile:
    """
    Free energy profile is a visualization of differences between local 
    minima points during metadynamics simulation. If the values seem 
    to converge to a mean value of the difference, it suggests, 
    but not fully proof, that the FES did converge to the correct shape.
    
    Command:
    ```python
    fep = metadynminer.FEProfile(minima, hillsfile)
    ```
    
    Parameters:
    
    * minima = metadynminer.Minima object
    
    * hillsfile = metadynminer.Hills object
    
    """
    def __init__(self, minima, hills):
        self.cvs = minima.cvs
        self.res = minima.res
        self.minima = minima.minima
        self.periodic = minima.periodic
        self.heights = hills.get_heights()
        
        if self.cvs >= 1:
            self.cv1_name = minima.cv1_name
            self.cv1min = minima.cv1min
            self.cv1max = minima.cv1max
            self.cv1 = hills.get_cv1()
            self.s1 = hills.get_sigma1()
            self.cv1per = hills.get_cv1per()
        if self.cvs >= 2:
            self.cv2min = minima.cv2min
            self.cv2max = minima.cv2max
            self.cv2_name = minima.cv2_name
            self.cv2 = hills.get_cv2()
            self.s2 = hills.get_sigma2()
            self.cv2per = hills.get_cv2per()
        if self.cvs == 3:
            self.cv3min = minima.cv3min
            self.cv3max = minima.cv3max
            self.cv3_name = minima.cv3_name
            self.cv3 = hills.get_cv3()
            self.s3 = hills.get_sigma3()
            self.cv3per = hills.get_cv3per()
        
        if len(minima.minima.shape)>1:
            self.makefeprofile(hills)
        else: 
            print("There is only one local minimum on the free energy surface.")
        
        
    def makefeprofile(self, hills):
        """
        Internal method to calculate free energy profile.
        """
        hillslenght = len(hills.get_cv1())
        
        if hillslenght < 256:
            profilelenght = hillslenght
            scantimes = np.array(range(hillslenght))
        else:
            profilelenght = 256
            scantimes = np.array(((hillslenght/(profilelenght))*np.array((range(1,profilelenght+1)))))
            scantimes -= 1
            scantimes = scantimes.astype(int)
        
        number_of_minima = self.minima.shape[0]
        
        self.feprofile = np.zeros((self.minima.Minimum.shape[0]+1))
        
        if self.cvs == 1:
            if self.periodic[0]:
                cv1min = self.cv1per[0]
                cv1max = self.cv1per[1]
                cv1_fes_range = np.abs(self.cv1per[1]-self.cv1per[0])
            else:
                cv1range = self.cv1max-self.cv1min
                cv1min = self.cv1min
                cv1max = self.cv1max
                cv1min -= cv1range*0.15          
                cv1max += cv1range*0.15
                cv1_fes_range = cv1max - cv1min
            
            fes = np.zeros((self.res))
            
            lasttime = 0
            line = 0
            for time in scantimes:
                for x in self.minima.iloc[:,3]:
                    dist_cv1 = self.cv1[lasttime:time]-float(x)
                    if self.periodic[0]:
                        dist_cv1[dist_cv1<-0.5*cv1_fes_range] += cv1_fes_range
                        dist_cv1[dist_cv1>+0.5*cv1_fes_range] -= cv1_fes_range

                    dp2 = dist_cv1**2/(2*self.s1[lasttime:time]**2)
                    tmp = np.zeros(self.cv1[lasttime:time].shape)
                    tmp[dp2<2.5] = self.heights[lasttime:time][dp2<2.5] * (np.exp(-dp2[dp2<2.5]) * 1.00193418799744762399 - 0.00193418799744762399)
                    fes[int((float(x)-cv1min)*self.res/cv1_fes_range)] -= tmp.sum()

                profileline = [time]
                for m in range(number_of_minima):
                    profileline.append(fes[int(float(self.minima.iloc[m,2]))]-\
                                       fes[int(float(self.minima.iloc[0,2]))])
                self.feprofile = np.vstack([self.feprofile, profileline])

                lasttime = time
            
        elif self.cvs == 2:
            if self.periodic[0]:
                cv1min = self.cv1per[0]
                cv1max = self.cv1per[1]
                cv1_fes_range = np.abs(self.cv1per[1]-self.cv1per[0])
            else:
                cv1range = self.cv1max-self.cv1min
                cv1min = self.cv1min
                cv1max = self.cv1max
                cv1min -= cv1range*0.15          
                cv1max += cv1range*0.15
                cv1_fes_range = cv1max - cv1min
                
            if self.periodic[1]:
                cv2min = self.cv2per[0]
                cv2max = self.cv2per[1]
                cv2_fes_range = np.abs(self.cv2per[1]-self.cv2per[0])
            else:
                cv2range = self.cv2max-self.cv2min
                cv2min = self.cv2min
                cv2max = self.cv2max
                cv2min -= cv2range*0.15          
                cv2max += cv2range*0.15
                cv2_fes_range = cv2max - cv2min
            
            fes = np.zeros((self.res, self.res))
            
            lasttime = 0
            line = 0
            for time in scantimes:
                for x in self.minima.iloc[:,4]:
                    dist_cv1 = self.cv1[lasttime:time]-float(x)
                    if self.periodic[0]:
                        dist_cv1[dist_cv1<-0.5*cv1_fes_range] += cv1_fes_range
                        dist_cv1[dist_cv1>+0.5*cv1_fes_range] -= cv1_fes_range
                    
                    for y in self.minima.iloc[:,5]:
                        dist_cv2 = self.cv2[lasttime:time]-float(y)
                        if self.periodic[1]:
                            dist_cv2[dist_cv2<-0.5*cv2_fes_range] += cv2_fes_range
                            dist_cv2[dist_cv2>+0.5*cv2_fes_range] -= cv2_fes_range
                    
                        dp2 = dist_cv1**2/(2*self.s1[lasttime:time]**2) + dist_cv2**2/(2*self.s2[lasttime:time]**2)
                        tmp = np.zeros(self.cv1[lasttime:time].shape)
                        tmp[dp2<6.25] = self.heights[lasttime:time][dp2<6.25] * (np.exp(-dp2[dp2<6.25]) * 1.00193418799744762399 - 0.00193418799744762399)
                        fes[int((float(x)-cv1min)*self.res/cv1_fes_range),int((float(y)-cv2min)*self.res/cv2_fes_range)] -= tmp.sum()
                
                # save profile
                profileline = [time]
                for m in range(number_of_minima):
                    profileline.append(fes[int(float(self.minima.iloc[m,2])),int(float(self.minima.iloc[m,3]))]-\
                                       fes[int(float(self.minima.iloc[0,2])),int(float(self.minima.iloc[0,3]))])
                self.feprofile = np.vstack([self.feprofile, profileline])

                lasttime = time
            
        elif self.cvs == 3:
            if self.periodic[0]:
                cv1min = self.cv1per[0]
                cv1max = self.cv1per[1]
                cv1_fes_range = np.abs(self.cv1per[1]-self.cv1per[0])
            else:
                cv1range = self.cv1max-self.cv1min
                cv1min = self.cv1min
                cv1max = self.cv1max
                cv1min -= cv1range*0.15          
                cv1max += cv1range*0.15
                cv1_fes_range = cv1max - cv1min
                
            if self.periodic[1]:
                cv2min = self.cv2per[0]
                cv2max = self.cv2per[1]
                cv2_fes_range = np.abs(self.cv2per[1]-self.cv2per[0])
            else:
                cv2range = self.cv2max-self.cv2min
                cv2min = self.cv2min
                cv2max = self.cv2max
                cv2min -= cv2range*0.15          
                cv2max += cv2range*0.15
                cv2_fes_range = cv2max - cv2min
                
            if self.periodic[3]:
                cv3min = self.cv3per[0]
                cv3max = self.cv3per[1]
                cv3_fes_range = np.abs(self.cv3per[1]-self.cv3per[0])
            else:
                cv3range = self.cv3max-self.cv3min
                cv3min = self.cv3min
                cv3max = self.cv3max
                cv3min -= cv3range*0.15          
                cv3max += cv3range*0.15
                cv3_fes_range = cv3max - cv3min
            
            fes = np.zeros((self.res, self.res, self.res))
            
            lasttime = 0
            line = 0
            for time in scantimes:
                for x in self.minima.iloc[:,5]:
                    dist_cv1 = self.cv1[lasttime:time]-float(x)
                    if self.periodic[0]:
                        dist_cv1[dist_cv1<-0.5*cv1_fes_range] += cv1_fes_range
                        dist_cv1[dist_cv1>+0.5*cv1_fes_range] -= cv1_fes_range
                    
                    for y in self.minima.iloc[:,6]:
                        dist_cv2 = self.cv2[lasttime:time]-float(y)
                        if self.periodic[1]:
                            dist_cv2[dist_cv2<-0.5*cv2_fes_range] += cv2_fes_range
                            dist_cv2[dist_cv2>+0.5*cv2_fes_range] -= cv2_fes_range
                        
                        for z in self.minima.iloc[:,7]:
                            dist_cv3 = self.cv3[lasttime:time]-float(z)
                            if self.periodic[2]:
                                dist_cv3[dist_cv3<-0.5*cv3_fes_range] += cv3_fes_range
                                dist_cv3[dist_cv3>+0.5*cv3_fes_range] -= cv3_fes_range
                    
                            dp2 = (dist_cv1**2/(2*self.s1[lasttime:time]**2) + 
                                   dist_cv2**2/(2*self.s2[lasttime:time]**2) + 
                                   dist_cv3**2/(2*self.s3[lasttime:time]**2))
                            tmp = np.zeros(self.cv1[lasttime:time].shape)
                            tmp[dp2<15.625] = (self.heights[lasttime:time][dp2<15.625] * 
                                               (np.exp(-dp2[dp2<15.625]) * 1.00193418799744762399 - 0.00193418799744762399))
                            fes[int((float(x)-cv1min)*self.res/cv1_fes_range),
                                int((float(y)-cv2min)*self.res/cv2_fes_range),
                                int((float(z)-cv3min)*self.res/cv3_fes_range)] -= tmp.sum()
                
                # save profile
                profileline = [time]
                for m in range(number_of_minima):
                    profileline.append(fes[int(float(self.minima.iloc[m,2])),
                                           int(float(self.minima.iloc[m,3])),
                                           int(float(self.minima.iloc[m,4]))]-\
                                       fes[int(float(self.minima.iloc[0,2])),
                                           int(float(self.minima.iloc[0,3])),
                                           int(float(self.minima.iloc[0,4]))])
                self.feprofile = np.vstack([self.feprofile, profileline])

                lasttime = time
            
        else:
            print("Fes object doesn't have supported number of CVs.")
    
    def plot(self, name="FEprofile.png",image_size=[10,7], xlabel=None, ylabel=None, label_size=12, cmap="jet"):
        """
        Visualization function for FEP. 
        
        ```python
        fep.plot()
        ```
        
        
        Parameters:
        
        
        * name (default="FEProfile.png") = name for .png file to save the plot to
        
        * image_size (default=[10,7]) = list of two dimensions of the picture
        
        * xlabel (default="time (ps)")
        
        * ylabel (default="free energy difference (kJ/mol)") 
        
        * label_size (default=12) = size of labels
        
        * cmap (default="jet") = matplotlib colormap used for coloring the line of the minima
        """
        plt.figure(figsize=(image_size[0],image_size[1]))
        
        cmap=cm.get_cmap(cmap)
        
        #colors = cm.jet((self.minima.iloc[:,1].to_numpy()).astype(float)/\
        #                (np.max(self.minima.iloc[:,1].to_numpy().astype(float))))
        colors = cmap(np.linspace(0,1,self.minima.shape[0]))
        for m in range(self.minima.shape[0]):
            plt.plot(self.feprofile[:,0], self.feprofile[:,m+1], color=colors[m])

        if xlabel == None:
            plt.xlabel('time (ps)', size=label_size)
        else:
            plt.xlabel(xlabel, size=label_size)
        if ylabel == None:
            plt.ylabel('free energy difference (kJ/mol)', size=label_size)
        else:
            plt.ylabel(ylabel, size=label_size)
        plt.savefig(name)