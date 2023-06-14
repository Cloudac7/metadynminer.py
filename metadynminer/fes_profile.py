import numpy as np
import pandas as pd

from itertools import product
from metadynminer.hills import Hills
from metadynminer.minima import Minima


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
    - minima = metadynminer.Minima object
    - hillsfile = metadynminer.Hills object

    """

    def __init__(self, minima: Minima, hills: Hills):
        self.cvs = minima.cvs
        self.res = minima.res

        if type(minima.minima) != pd.DataFrame:
            raise ValueError(
                "There is only one local minimum on the free energy surface."
            )
        self.minima = minima.minima

        self.periodic = minima.periodic
        self.heights = hills.get_heights()

        self.cv_name = minima.cv_name
        self.cv_min = minima.cv_min
        self.cv_max = minima.cv_max
        self.cv_per = minima.cv_per
        self.sigma = hills.sigma
        self.cv = hills.cv

        self.makefeprofile(hills)

    def makefeprofile(self, hills):
        """
        Internal method to calculate free energy profile.
        """
        hillslenght = len(hills.cv[:, 0])

        if hillslenght < 256:
            profilelenght = hillslenght
            scantimes = np.array(range(hillslenght))
        else:
            profilelenght = 256
            scantimes = np.array(
                ((hillslenght/(profilelenght))*np.array((range(1, profilelenght+1)))))
            scantimes -= 1
            scantimes = scantimes.astype(int)

        number_of_minima = self.minima.shape[0]

        self.feprofile = np.zeros((self.minima["Minimum"].shape[0]+1))

        cvs = self.cvs
        cv_min, cv_max = self.cv_min, self.cv_max
        cv_fes_range = self.cv_max - self.cv_min

        fes = np.zeros((self.res, self.res))

        lasttime = 0
        line = 0
        for time in scantimes:
            minima_cv_matrix = np.array(
                self.minima.iloc[:, cvs+2:2*cvs+2], dtype=float
            )
            for coords in product(*minima_cv_matrix.T):
                dist_cvs = []
                for i in range(cvs):
                    dist_cv = self.cv[:, i][lasttime:time] - coords[i]
                    if self.periodic[i]:
                        dist_cv[dist_cv < -0.5*cv_fes_range[i]
                                ] += cv_fes_range[i]
                        dist_cv[dist_cv > +0.5*cv_fes_range[i]
                                ] -= cv_fes_range[i]
                    dist_cvs.append(dist_cv)
                dp2 = np.sum(np.array(dist_cvs)**2, axis=0) / \
                    (2*self.sigma[:, 0][lasttime:time]**2)
                tmp = np.zeros(self.cv[:, 0][lasttime:time].shape)
                tmp[dp2 < 2.5] = self.heights[lasttime:time][dp2 < 2.5] * \
                    (np.exp(-dp2[dp2 < 2.5]) *
                     1.00193418799744762399 - 0.00193418799744762399)
                fes[tuple([
                    int((float(coords[i])-cv_min[i])*self.res/cv_fes_range[i]) for i in range(cvs)
                ])] -= tmp.sum()

            # save profile
            profileline = [time]
            for m in range(number_of_minima):
                profile = fes[tuple([
                    int(float(self.minima.iloc[m, i+2])) for i in range(cvs)
                ])] - fes[tuple([
                    int(float(self.minima.iloc[0, i+2])) for i in range(cvs)
                ])]
                profileline.append(profile)
            self.feprofile = np.vstack([self.feprofile, profileline])

            lasttime = time

    def plot(self, name="FEprofile.png", image_size=[10, 7], xlabel=None, ylabel=None, label_size=12, cmap="jet"):
        """
        Visualization function for FEP. 

        ```python
        fep.plot()
        ```

        Parameters:

        - name (default="FEProfile.png") = name for .png file to save the plot to
        - image_size (default=[10,7]) = list of two dimensions of the picture
        - xlabel (default="time (ps)")
        - ylabel (default="free energy difference (kJ/mol)") 
        - label_size (default=12) = size of labels
        - cmap (default="jet") = matplotlib colormap used for coloring the line of the minima
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        plt.figure(figsize=(image_size[0], image_size[1]))

        cmap = cm.get_cmap(cmap)

        # colors = cm.jet((self.minima.iloc[:,1].to_numpy()).astype(float)/\
        #                (np.max(self.minima.iloc[:,1].to_numpy().astype(float))))
        colors = cmap(np.linspace(0, 1, self.minima.shape[0]))
        for m in range(self.minima.shape[0]):
            plt.plot(self.feprofile[:, 0],
                     self.feprofile[:, m+1], color=colors[m])

        if xlabel == None:
            plt.xlabel('time (ps)', size=label_size)
        else:
            plt.xlabel(xlabel, size=label_size)
        if ylabel == None:
            plt.ylabel('free energy difference (kJ/mol)', size=label_size)
        else:
            plt.ylabel(ylabel, size=label_size)
        plt.savefig(name)
