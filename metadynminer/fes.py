import sys
import numpy as np

from tqdm import tqdm, trange
from typing import Optional, List
from metadynminer.hills import Hills


class Fes:
    """
    Object of this class is created to compute the free energy surface corresponding to the provided Hills object. 
    Command:
    ```python
    fes = metadynminer.Fes(hills=hillsfile)
    ```
    parameters:

    - hills = Hills object

    - resolution (default=256) = \
        should be positive integer, controls the resolution of FES

    - original (default=False) = \
        boolean, if False, FES will be calculated using very fast, but not
        'exact' Bias Sum Algorithm
        if True, FES will be calculated with slower algorithm, 
        but it will be exactly the same as FES calculated 
        with PLUMED sum_hills function

    - cv_selected (default=None) = \
        list of integers, specifying which CVs should be used for FES calculation

    - cv1range, cv2range, cv3range = \
        lists of two numbers, defining lower and upper bound of 
        the respective CV (in the units of the CVs)
    """

    def __init__(
            self,
            hills: Hills, 
            original: bool = False,
            calculate_new_fes: bool = True,
            resolution: int = 256, 
            cv_range: Optional[List[float]] = None, 
            cv_select: Optional[List[int]] = None,
            time_min: int = 0,
            time_max: Optional[int] = None
        ):
        self.res = resolution
        self.fes = None

        self.hills = hills
        self.periodic = hills.get_periodic()
        
        if cv_select != None:
            self.cvs = len(cv_select)
        else:
            cv_select = [i for i in range(self.cvs)]

        if time_max != None:
            if time_max <= time_min:
                print("Error: End time is lower than start time")
            if time_max > len(self.hills.cv[:, 0]):
                time_max = len(self.hills.cv[:, 0])
                print(f"Error: End time {time_max} is higher than",
                      f"number of lines in HILLS file {len(self.hills.cv[:, 0])},",
                      "which will be used instead.")

        if calculate_new_fes:
            if not original:
                self.makefes(resolution, cv_select, cv_range, time_min, time_max)
            else:
                self.makefes2(resolution, cv_select, cv_range, time_min, time_max)

    def generate_cv_map(
            self, 
            cv_select: Optional[List[int]] = None,
            cv_range: Optional[List[float]] = None 
        ):
        """generate CV map

        Args:
            cv_range (Optional[List[float]], optional): The range of CV.
            cv_indexes (Optional[List[int]], optional): \
                Indexes of CV to be sum. Defaults to None.

        Returns:
            Tuple: cv_min, cv_max, cv_fes_range
        """
        if cv_select is None:
            cv_select = list(range(self.cvs))

        if cv_range is None:
            cv_min = self.hills.cv_min[cv_select]
            cv_max = self.hills.cv_max[cv_select]
            cv_range = self.hills.cv_max - self.hills.cv_min
        else:
            cv_min = np.full(len(cv_select), cv_range[0])
            cv_max = np.full(len(cv_select), cv_range[1])
        self.cv_range = cv_range

        cv_min[~self.periodic[cv_select]] -= cv_min[~self.periodic[cv_select]] * 0.15
        cv_max[~self.periodic[cv_select]] += cv_min[~self.periodic[cv_select]] * 0.15
        cv_fes_range = np.abs(cv_max - cv_min)

        self.cv_min = cv_min
        self.cv_max = cv_max
        self.cv_fes_range = cv_fes_range

    def makefes(
            self, 
            resolution: Optional[int] = None, 
            cv_select: Optional[List[int]] = None,
            cv_range: Optional[List[float]] = None, 
            time_min: Optional[int] = None,
            time_max: Optional[int] = None
        ):
        """calculate FES using fast algorithm"""

        if resolution is None:
            resolution = self.res

        if cv_select is None:
            cv_select = list(range(self.cvs))

        self.cv_select = cv_select
        self.generate_cv_map(cv_select, cv_range)
        cv_min = self.cv_min
        cv_max = self.cv_max
        cv_fes_range = self.cv_fes_range

        cv_bins = np.array([np.ceil(
            (self.hills.cv[time_min:time_max, cv_index] - cv_min[cv_index]) * resolution / cv_fes_range[cv_index]
        ) for cv_index in cv_select]) 
        cvs = len(cv_select)
        cv_bins = cv_bins.astype(int)

        sigma = np.array([self.hills.sigma[cv_index][0] for cv_index in cv_select])
        sigma_res = (sigma * self.res) / (cv_max - cv_min)

        gauss_res = np.max((8 * sigma_res).astype(int))
        if gauss_res % 2 == 0:
            gauss_res += 1
    
        gauss_center_to_end = int((gauss_res - 1) / 2)
        gauss_center = gauss_center_to_end + 1
        grids = np.meshgrid(*[np.arange(gauss_res)] * cvs)
        exponent = 0.
        for i in range(len(sigma_res)):
            i_diff = (grids[i] + 1) - gauss_center
            exponent += -(i_diff ** 2) / (2 * sigma_res[i] ** 2)
        gauss = -np.exp(exponent)

        fes = np.zeros([resolution] * cvs)
        for line in trange(len(cv_bins[0]), desc="Constructing FES"):
            # create a meshgrid of the indexes of the fes that need to be edited
            # size of the meshgrid is the same as the size of the gauss
            fes_index_to_edit = np.meshgrid(
                *([
                    np.arange(gauss_res) - gauss_center
                ] * len(cv_bins[:, line]))
            )

            # create a mask to avoid editing indexes outside the fes
            local_mask = np.ones_like(gauss, dtype=int)
            for d in range(cvs):
                fes_index_to_edit[d] += cv_bins[d][line]
                if not self.periodic[d]:
                    mask = np.where(
                        (fes_index_to_edit[d] < 0) & (fes_index_to_edit[d] > resolution - 1)
                    )[0]
                    # if the cv is not periodic, remove the indexes outside the fes
                    local_mask[mask] = 0
                # make sure the indexes are inside the fes
                fes_index_to_edit[d] = np.mod(fes_index_to_edit[d], resolution)
            fes[tuple(fes_index_to_edit)] += gauss * local_mask * self.hills.heights[line]
        fes = fes - np.min(fes)
        self.fes = fes
        return fes

    def makefes2(
            self, 
            resolution: Optional[int], 
            cv_select: Optional[List[int]] = None, 
            cv_range: Optional[List[float]] = None, 
            time_min: Optional[int] = None,
            time_max: Optional[int] = None
        ):
        """
        Function internally used to sum Hills in the same way as Plumed sum_hills. 
        """

        if resolution is None:
            resolution = self.res
        
        self.generate_cv_map(cv_select, cv_range)
        cv_min = self.cv_min
        cv_max = self.cv_max
        cv_fes_range = self.cv_fes_range
        
        cvs = len(self.cv_select)
        fes = np.zeros([resolution] * cvs)
        if time_min and time_max:
            time_limit = time_max - time_min
        else:
            time_min = 0
            time_max = self.hills.cv[:, 0].shape[0]
            time_limit = time_max - time_min

        for index in tqdm(np.ndindex(fes.shape), # type: ignore
                          desc="Constructing FES",
                          total=np.prod(fes.shape)): # type: ignore
            dp2_array = np.zeros([cvs, time_limit])
            for i, cv_idx in enumerate(self.cv_select):
                dist_cv = \
                    self.hills.cv[:, cv_idx] - (cv_min[i] + index[i] * cv_fes_range[i] / resolution)
                if self.periodic[cv_idx]:
                    dist_cv[dist_cv < -0.5*cv_fes_range[i]] += cv_fes_range[i]
                    dist_cv[dist_cv > +0.5*cv_fes_range[i]] -= cv_fes_range[i]
                dp2_local = dist_cv ** 2 / (2 * self.hills.sigma[cv_idx][0] ** 2)
                dp2_array[i] = dp2_local
            dp2 = np.sum(dp2_array, axis=0)

            tmp = np.zeros(time_limit)
            tmp[dp2 < 6.25] = self.hills.heights[dp2 < 6.25] * \
                    (np.exp(-dp2[dp2 < 6.25]) *
                     1.00193418799744762399 - 0.00193418799744762399)
            fes[index] = -tmp.sum()

        fes = fes - np.min(fes)
        self.fes = np.array(fes)

    def plot(self, png_name=None, contours=True, contours_spacing=0.0, aspect=1.0, cmap="jet",
             energy_unit="kJ/mol", xlabel=None, ylabel=None, zlabel=None, label_size=12, image_size=[10, 7],
             vmin=0, vmax=None, opacity=0.2, levels=None):
        """
        Function used to visualize FES, based on Matplotlib and PyVista. 

        ```python
        fes.plot()
        ```

        Parameters:

        - png_name = String. If this parameter is supplied, the picture of FES will be saved under this name to the current working directory.
        - contours (default=True) = whether contours should be shown on 2D FES
        - contours_spacing (default=0.0) = when a positive number is set, it will be used as spacing for contours on 2D FES. 
                Otherwise, if contours=True, there will be five equally spaced contour levels.
        - aspect (default = 1.0) = aspect ratio of the graph. Works with 1D and 2D FES. 
        - cmap (default = "jet") = Matplotlib colormap used to color 2D or 3D FES
        - energy_unit (default="kJ/mol") = String, used in description of colorbar
        - xlabel, ylabel, zlabel = Strings, if provided, they will be used as labels for the graphs
        - labelsize (default = 12) = size of text in labels
        - image_size (default = [10,7]) = List of the width and height of the picture
        - vmin (default=0) = real number, lower bound for the colormap on 2D FES
        - vmax = real number, upper bound for the colormap on 2D FES
        - opacity (default=0.2) = number between 0 and 1, is the opacity of isosurfaces of 3D FES
        - levels = Here you can specify list of free energy values for isosurfaces on 3D FES. 
                        If not provided, default values from contours parameters will be used instead. 
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        if self.fes is None:
            raise ValueError("FES not calculated yet. Use makefes() or makefes2() first.")

        if vmax == None:
            # if the addition is smaller than 0.01, the 3d plot stops working.
            vmax = np.max(self.fes) + 0.01

        if contours_spacing == 0.0:
            contours_spacing = (vmax-vmin)/5.0

        cmap = cm.get_cmap(cmap)

        cmap.set_over("white")
        cmap.set_under("white")
        
        cvs = len(self.cv_select)

        if cvs == 1:
            cv_index = self.cv_select[0]
            plt.figure(figsize=(image_size[cv_index], image_size[1]))
            X = np.linspace(self.cv_min[cv_index], self.cv_max[cv_index], self.res)
            plt.plot(X, self.fes)
            if xlabel == None:
                plt.xlabel(f'CV1 - {self.hills.cv_name[cv_index]}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(f'free energy ({energy_unit})', size=label_size)
            else:
                plt.ylabel(ylabel, size=label_size)

        elif cvs == 2:
            cv1_index = self.cv_select[0]
            cv2_index = self.cv_select[1]
            cv1min = self.cv_min[self.cv_select[0]]
            cv1max = self.cv_max[self.cv_select[0]]
            cv2min = self.cv_min[self.cv_select[1]]
            cv2max = self.cv_max[self.cv_select[1]]

            fig = plt.figure(figsize=(image_size[0], image_size[1]))
            plt.imshow(np.rot90(self.fes, axes=(0, 1)), cmap=cmap, interpolation='nearest',
                       extent=[cv1min, cv1max, cv2min, cv2max],
                       aspect=(((cv1max-cv1min)/(cv2max-cv2min))/(aspect)),
                       vmin=vmin, vmax=vmax) # type: ignore
            cbar = plt.colorbar()
            cbar.set_label(energy_unit, size=label_size)
            if contours:
                cont = plt.contour(np.rot90(self.fes, axes=(0, 1)),
                                   levels=np.arange(
                                       0, (vmax - 0.01), contours_spacing),
                                   extent=[cv1min, cv1max, cv2max, cv2min],
                                   colors="k")
                plt.clabel(cont, levels=np.arange(
                    0, (vmax - 0.01), contours_spacing))
            if xlabel == None:
                plt.xlabel(f'CV1 - {self.hills.cv_name[cv1_index]}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(f'CV2 - {self.hills.cv_name[cv1_index]}', size=label_size)
            else:
                plt.ylabel(ylabel, size=label_size)

        elif self.cvs == 3:
            try:
                import pyvista as pv
            except (ImportError, ModuleNotFoundError) as e:
                print(e)
                sys.exit(1)

            cv1_index = self.cv_select[0]
            cv2_index = self.cv_select[1]
            cv3_index = self.cv_select[2]
            cv1min = self.cv_min[self.cv_select[0]]
            cv1max = self.cv_max[self.cv_select[0]]
            cv2min = self.cv_min[self.cv_select[1]]
            cv2max = self.cv_max[self.cv_select[1]]
            cv3min = self.cv_min[self.cv_select[2]]
            cv3max = self.cv_max[self.cv_select[2]]
            if xlabel == None:
                xlabel = "CV1 - " + self.hills.cv_name[cv1_index]
            if ylabel == None:
                ylabel = "CV2 - " + self.hills.cv_name[cv2_index]
            if zlabel == None:
                zlabel = "CV3 - " + self.hills.cv_name[cv3_index]

            grid = pv.UniformGrid(
                dimensions=(self.res, self.res, self.res),
                spacing=((cv1max-cv1min)/self.res, (cv2max-cv2min) /
                         self.res, (cv3max-cv3min)/self.res),
                origin=(cv1min, cv2min, cv3min)
            )
            grid["vol"] = self.fes.ravel(order="F")
            if levels == None:
                contours = grid.contour(
                    np.arange(0, (vmax - 0.01), contours_spacing))
            else:
                contours = grid.contour(levels)
            fescolors = []
            for i in range(contours.points.shape[0]):
                fescolors.append(self.fes[int((contours.points[i, 0]-cv1min)*self.res/(cv1max-cv1min)),
                                          int((
                                              contours.points[i, 1]-cv2min)*self.res/(cv2max-cv2min)),
                                          int((contours.points[i, 2]-cv3min)*self.res/(cv3max-cv3min))])
            # %% Visualization
            pv.set_plot_theme('document')
            p = pv.Plotter()
            p.add_mesh(contours, scalars=fescolors, opacity=opacity,
                       cmap=cmap, show_scalar_bar=False, interpolate_before_map=True)
            p.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            p.show()

        else:
            raise ValueError("Only 1D, 2D and 3D FES are supported.")

        if png_name != None:
            plt.savefig(png_name)

    def set_fes(self, fes):
        self.fes = fes

    def surface_plot(self, cv_select=None, cv_range=None, cmap="jet",
                     energy_unit="kJ/mol", xlabel=None, ylabel=None, zlabel=None,
                     label_size=12, image_size=[12, 7], rstride=1, cstride=1, vmin=0, vmax=None):
        """
        Function for visualization of 2D FES as 3D surface plot. For now, it is based on Matplotlib, but there are issues with interactivity. 

        It can be interacted with in jupyter notebook or jupyter lab in %matplotlib widget mode. Otherwise it is just static image of the 3D surface plot. 

        ```python
        %matplotlib widget
        fes.surface_plot()
        ```

        There are future plans to implement this function using PyVista. 
        Hovewer, in current version of PyVista (0.38.5) there is an issue that labels on the 3rd axis for free energy are showing wrong values. 
        """
        import matplotlib.pyplot as plt

        if cv_select is None:
            cv_select = list(range(self.cvs))

        if self.cvs == 2:
            cv1_index = self.cv_select[0]
            cv2_index = self.cv_select[1]
            cv1min = self.cv_min[self.cv_select[0]]
            cv1max = self.cv_max[self.cv_select[0]]
            cv2min = self.cv_min[self.cv_select[1]]
            cv2max = self.cv_max[self.cv_select[1]]

            x = np.linspace(cv1min, cv1max, self.res)
            y = np.linspace(cv2min, cv2max, self.res)

            X, Y = np.meshgrid(x, y)
            Z = self.fes

            #grid = pv.StructuredGrid(X, Y, Z)
            # grid.plot()

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X, Y, Z, cmap=cmap, # type: ignore
                            rstride=rstride, cstride=cstride)

            if xlabel == None:
                ax.set_xlabel(f'CV1 - {self.hills.cv_name[cv1_index]}', size=label_size)
            else:
                ax.set_xlabel(xlabel, size=label_size)
            if ylabel == None:
                ax.set_ylabel(f'CV2 - {self.hills.cv_name[cv2_index]}', size=label_size)
            else:
                ax.set_ylabel(ylabel, size=label_size)
            if zlabel == None:
                ax.set_zlabel(f'free energy ({energy_unit})', size=label_size) # type: ignore
            else:
                ax.set_zlabel(zlabel, size=label_size) # type: ignore
        else:
            raise ValueError(
                f"Surface plot only works for FES with exactly two CVs, and this FES has {self.hills.cvs}"
            )

    def removeCV(
            self, 
            CV: int, 
            energy_unit: str = "kJ/mol", 
            kb: Optional[float] = None,
            temp: float = 300.0
        ):
        """
        This function is used to remove a CV from an existing FES. The function first recalculates the FES to an array of probabilities. The probabilities 
        are summed along the CV to be removed, and resulting probability distribution with 1 less dimension is converted back to FES. 

        Interactivity was working in jupyter notebook/lab with "%matplotlib widget".

        Parameters:

        - CV = integer, the index of CV to be removed. 
        - energy_unit (default="kJ/mol") = has to be either "kJ/mol" or "kcal/mol"
        - temp (default=300) = temperature of the simulation in Kelvins.
        """
        print(f"Removing CV {CV}.")

        if self.fes is None:
            raise ValueError("FES not calculated yet. Use makefes() or makefes2() first.")

        if CV > self.hills.cvs:
            print("Error: The CV to remove is not available in this FES object.")
            return None
        if kb == None:
            if energy_unit == "kJ/mol":
                kb = 8.314e-3
            elif energy_unit == "kcal/mol":
                kb = 8.314e-3 / 4.184
            else:
                raise ValueError("Please give the Boltzmann Constant in the energy unit.")

        if self.cvs == 1:
            print("Error: You can not remove the only CV. ")
            return None
        else:
            probabilities = np.exp(-self.fes / (kb * temp))
            new_prob = np.sum(probabilities, axis=CV)

            new_fes = Fes(hills=self.hills, calculate_new_fes=False)
            new_fes.fes = - kb * temp * np.log(new_prob)
            new_fes.fes = new_fes.fes - np.min(new_fes.fes)
            new_fes.res = self.res

            mask = np.ones(len(self.cv_select), dtype=bool)
            mask[CV] = False
            new_fes.cv_select = self.cv_select[mask]
            new_fes.cv_min = self.cv_min[mask]
            new_fes.cv_max = self.cv_max[mask]
            new_fes.cv_fes_range = self.cv_fes_range[mask]
            return new_fes

    def make_gif(self, gif_name="FES.gif", cmap="jet",
                 xlabel=None, ylabel=None, zlabel=None, label_size=12, image_size=[10, 7],
                 opacity=0.2, levels=None, frames=64):
        """
        Function that generates animation of 3D FES showing different isosurfaces.

        ```python
        fes.make_gif()
        ```

        Parameters:

        - gif_name (default="FES.gif") = String. Name of the gif of FES that will be saved in the current working directory.
        - cmap (default = "jet") = Matplotlib colormap used to color the 3D FES
        - xlabel, ylabel, zlabel = Strings, if provided, they will be used as labels for the graph
        - labelsize (default = 12) = size of text in labels
        - image_size (default = [10,7]) = List of the width and height of the picture
        - opacity (default = 0.2) = number between 0 and 1, is the opacity of isosurfaces of 3D FES
        - levels = Here you can specify list of free energy values for isosurfaces on 3D FES. 
                If not provided, default values from contours parameters will be used instead. 
        - frames (default = 64) = Number of frames the animation will be made of. 
        """
        try:
            import pyvista as pv
        except (ImportError, ModuleNotFoundError) as e:
            print(e)
            sys.exit(1)

        if self.fes is None:
            raise ValueError("FES not calculated yet. Use makefes() or makefes2() first.")

        if self.cvs == 3:
            cv1_index = self.cv_select[0]
            cv2_index = self.cv_select[1]
            cv3_index = self.cv_select[2]
            cv1min = self.cv_min[self.cv_select[0]]
            cv1max = self.cv_max[self.cv_select[0]]
            cv2min = self.cv_min[self.cv_select[1]]
            cv2max = self.cv_max[self.cv_select[1]]
            cv3min = self.cv_min[self.cv_select[2]]
            cv3max = self.cv_max[self.cv_select[2]]

            values = np.linspace(np.min(self.fes)+0.01,
                                 np.max(self.fes), num=frames)
            grid = pv.UniformGrid(
                dimensions=(self.res, self.res, self.res),
                spacing=((cv1max-cv1min)/self.res, (cv2max-cv2min) /
                         self.res, (cv3max-cv3min)/self.res),
                origin=(cv1min, cv2min, cv3min),
            )
            grid["vol"] = np.ravel(self.fes, order="F")
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
                plotter.show_grid(
                    xlabel=f"CV1 - {self.hills.cv_name[cv1_index]}", 
                    ylabel=f"CV2 - {self.hills.cv_name[cv2_index]}", 
                    zlabel=f"CV3 - {self.hills.cv_name[cv3_index]}")
            else:
                plotter.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
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
            raise ValueError("Error: gif_plot is only available for FES with 3 CVs.")
