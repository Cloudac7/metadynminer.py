import sys
import numpy as np

from tqdm import tqdm, trange
from typing import Optional, List
from metadynminer.hills import Hills


class Fes:
    """
    Computes the free energy surface corresponding to the provided Hills object.

    Usage:
    ```python
    fes = metadynminer.Fes(hills=hillsfile)
    ```

    Args:
        hills (Hills): The Hills object used for computing the free energy surface.
        original (bool, optional): \
            If False, the free energy surface will be calculated using a fast but approximate algorithm. \
            If True, it will be calculated using a slower but exact algorithm \
            (same as FES calculated with PLUMED `sum_hills` function). \
            Defaults to False.
        calculate_new_fes (bool, optional): \
            If True, the free energy surface will be calculated to form `self.fes`. \
            Defaults to True.
        resolution (int, optional): \
            The resolution of the free energy surface. Defaults to 256.
        cv_select (List[int], optional): \
            A list of integers specifying which collective variables (CVs) should be used for FES calculation. \
            Defaults to None.
        cv_range (List[float], optional): \
            A list of two numbers defining the lower and upper bounds of the CVs (in the units of the CV). \
            Defaults to None. \
        time_min (int): The starting time step of simulation. Defaults to 0.
        time_max (int, optional): The ending time step of simulation. Defaults to None.
    """


    def __init__(
        self,
        hills: Hills,
        original: bool = False,
        calculate_new_fes: bool = True,
        resolution: int = 256,
        cv_select: Optional[List[int]] = None,
        cv_range: Optional[List[float]] = None,
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
            cv_select = [i for i in range(self.hills.cvs)]

        if time_max != None:
            if time_max <= time_min:
                print("Error: End time is lower than start time")
            if time_max > len(self.hills.cv[:, 0]):
                time_max = len(self.hills.cv[:, 0])
                print(
                    f"Error: End time {time_max} is higher than",
                    f"number of lines in HILLS file {len(self.hills.cv[:, 0])},",
                    "which will be used instead."
                )

        if calculate_new_fes:
            if not original:
                self.makefes(resolution, cv_select,
                             cv_range, time_min, time_max)
            else:
                self.makefes2(resolution, cv_select,
                              cv_range, time_min, time_max)

    def generate_cv_map(
        self,
        cv_select: Optional[List[int]] = None,
        cv_range: Optional[List[float]] = None
    ):
        """generate CV map

        Args:
            cv_select (List[int], optional): \
                A list of integers specifying which collective variables (CVs) should be used for FES calculation. \
                Defaults to None.
            cv_range (Optional[List[float]], optional): \
                A list of two numbers defining the lower and upper bounds of the CVs (in the units of the CV). \
                Defaults to None. \

        Returns:
            Tuple: cv_min, cv_max, cv_fes_range
        """
        if cv_select is None:
            cv_select = list(range(self.cvs))
        
        self.cv_select = cv_select

        if cv_range is None:
            cv_min = self.hills.cv_min[cv_select]
            cv_max = self.hills.cv_max[cv_select]
            cv_range = self.hills.cv_max - self.hills.cv_min
        else:
            if len(cv_range) != len(cv_select) and len(cv_range) != 2:
                raise ValueError(
                    "cv_range has to have the same length as cv_select"
                    "or be a list of two numbers"
                )
            elif len(cv_range) == 2:
                if type(cv_range[0]) == int:
                    cv_min = np.full(len(cv_select), cv_range[0])
                    cv_max = np.full(len(cv_select), cv_range[1])
            else:
                cv_min = np.array(cv_range)[:, 0]
                cv_max = np.array(cv_range)[:, 1]

        self.cv_range = cv_range

        cv_min[~self.periodic[cv_select] # type: ignore
               ] -= cv_range[~self.periodic[cv_select]] * 0.15 # type: ignore
        cv_max[~self.periodic[cv_select] # type: ignore
               ] += cv_range[~self.periodic[cv_select]] * 0.15 # type: ignore
        cv_fes_range = np.abs(cv_max - cv_min) # type: ignore

        # generate remapped cv_min and cv_max
        self.cv_min = cv_min # type: ignore
        self.cv_max = cv_max # type: ignore
        self.cv_fes_range = cv_fes_range

    def makefes(
        self,
        resolution: Optional[int] = None,
        cv_select: Optional[List[int]] = None,
        cv_range: Optional[List[float]] = None,
        time_min: Optional[int] = None,
        time_max: Optional[int] = None
    ):
        """Function used internally for summing hills in Hills object with the fast Bias Sum Algorithm. 

        Args:
            resolution (int, optional): \
                The resolution of the free energy surface. Defaults to 256.
            cv_select (List[int], optional): \
                A list of integers specifying which collective variables (CVs) should be used for FES calculation. \
                Defaults to None.
            cv_range (List[float], optional): \
                A list of two numbers defining the lower and upper bounds of the CVs (in the units of the CV). \
                Defaults to None. \
            time_min (int): The starting time step of simulation. Defaults to 0.
            time_max (int, optional): The ending time step of simulation. Defaults to None.
        """

        if resolution is None:
            resolution = self.res
        self.generate_cv_map(cv_select, cv_range)

        cv_min = self.cv_min
        cv_max = self.cv_max
        cv_fes_range = self.cv_fes_range

        cv_bins = np.array([np.ceil(
            (self.hills.cv[time_min:time_max, cv_index] -
             cv_min[cv_index]) * resolution / cv_fes_range[cv_index]
        ) for cv_index in cv_select])
        cvs = len(cv_select)
        cv_bins = cv_bins.astype(int)

        sigma = np.array([self.hills.sigma[cv_index][0]
                         for cv_index in cv_select])
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
                        (fes_index_to_edit[d] < 0) & (
                            fes_index_to_edit[d] > resolution - 1)
                    )[0]
                    # if the cv is not periodic, remove the indexes outside the fes
                    local_mask[mask] = 0
                # make sure the indexes are inside the fes
                fes_index_to_edit[d] = np.mod(fes_index_to_edit[d], resolution)
            fes[tuple(fes_index_to_edit)] += gauss * \
                local_mask * self.hills.heights[line]
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
        Function internally used to sum Hills in the same way as Plumed `sum_hills`. 

        Args:
            resolution (int, optional): \
                The resolution of the free energy surface. Defaults to 256.
            cv_select (List[int], optional): \
                A list of integers specifying which collective variables (CVs) should be used for FES calculation. \
                Defaults to None.
            cv_range (List[float], optional): \
                A list of two numbers defining the lower and upper bounds of the CVs (in the units of the CV). \
                Defaults to None. \
            time_min (int): The starting time step of simulation. Defaults to 0.
            time_max (int, optional): The ending time step of simulation. Defaults to None.
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

        for index in tqdm(np.ndindex(fes.shape),  # type: ignore
                          desc="Constructing FES",
                          total=np.prod(fes.shape)):  # type: ignore
            dp2_array = np.zeros([cvs, time_limit])
            for i, cv_idx in enumerate(self.cv_select):
                dist_cv = \
                    self.hills.cv[:, cv_idx] - \
                    (cv_min[i] + index[i] * cv_fes_range[i] / resolution)
                if self.periodic[cv_idx]:
                    dist_cv[dist_cv < -0.5*cv_fes_range[i]] += cv_fes_range[i]
                    dist_cv[dist_cv > +0.5*cv_fes_range[i]] -= cv_fes_range[i]
                dp2_local = dist_cv ** 2 / \
                    (2 * self.hills.sigma[cv_idx][0] ** 2)
                dp2_array[i] = dp2_local
            dp2 = np.sum(dp2_array, axis=0)

            tmp = np.zeros(time_limit)
            tmp[dp2 < 6.25] = self.hills.heights[dp2 < 6.25] * \
                (np.exp(-dp2[dp2 < 6.25]) *
                 1.00193418799744762399 - 0.00193418799744762399)
            fes[index] = -tmp.sum()

        fes = fes - np.min(fes)
        self.fes = np.array(fes)

    def plot(
        self,
        png_name: Optional[str] = None,
        contours: bool = True,
        contours_spacing: float = 0.0,
        aspect: float = 1.0,
        cmap: str = "jet",
        energy_unit: str = "kJ/mol",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        label_size: int = 12,
        image_size: List[int] = [10, 7],
        vmin: float = 0,
        vmax: Optional[float] = None,
        opacity: float = 0.2,
        levels: Optional[List[float]] = None,
    ):
        """
        Visualizes the free energy surface (FES) using Matplotlib and PyVista.

        Usage:
        ```python
        fes.plot()
        ```

        Args:
            png_name (str, optional): If provided, the picture of FES will be saved under this name in the current working directory.
            contours (bool, default=True): Determines whether contours should be shown on the 2D FES.
            contours_spacing (float, default=0.0): When a positive number is set, it will be used as the spacing for contours on the 2D FES. Otherwise, if contours=True, there will be five equally spaced contour levels.
            aspect (float, default=1.0): The aspect ratio of the graph. Works with 1D and 2D FES.
            cmap (str, default="jet"): The Matplotlib colormap used to color the 2D or 3D FES.
            energy_unit (str, default="kJ/mol"): The unit used in the description of the colorbar.
            xlabel, ylabel, zlabel (str, optional): If provided, they will be used as labels for the graphs.
            label_size (int, default=12): The size of text in the labels.
            image_size (List[int], default=[10,7]): The width and height of the picture.
            vmin (float, default=0): The lower bound for the colormap on the 2D FES.
            vmax (float, optional): The upper bound for the colormap on the 2D FES.
            opacity (float, default=0.2): A number between 0 and 1 that represents the opacity of isosurfaces in the 3D FES.
            levels (List[float], optional): A list of free energy values for isosurfaces in the 3D FES. If not provided, default values from the contours parameters will be used instead.
        """

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

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
            X = np.linspace(self.cv_min[cv_index],
                            self.cv_max[cv_index], self.res)
            plt.plot(X, self.fes)
            if xlabel == None:
                plt.xlabel(
                    f'CV1 - {self.hills.cv_name[cv_index]}', size=label_size)
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
                       vmin=vmin, vmax=vmax)  # type: ignore
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
                plt.xlabel(
                    f'CV1 - {self.hills.cv_name[cv1_index]}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(
                    f'CV2 - {self.hills.cv_name[cv2_index]}', size=label_size)
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

    def surface_plot(
        self,
        cv_select: Optional[None] = None,
        cv_range: Optional[None] = None,
        cmap: str = "jet",
        energy_unit: str = "kJ/mol",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        label_size: int = 12,
        image_size: List[int] = [12, 7],
        rstride: int = 1,
        cstride: int = 1,
        vmin: int = 0,
        vmax: Optional[int] = None,
    ):
        """
        Visualizes the 2D free energy surface (FES) as a 3D surface plot using Matplotlib.

        Note: Interactivity is currently limited to jupyter notebook or jupyter lab in `%matplotlib widget` mode. Otherwise, it is a static image of the 3D surface plot.

        Usage:
        ```python
        %matplotlib widget
        fes.surface_plot()
        ```

        Future plans include implementing this function using PyVista. However, in the current version of PyVista (0.38.5), there is an issue with labels on the 3rd axis for free energy showing wrong values.
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
            ax.plot_surface(X, Y, Z, cmap=cmap,  # type: ignore
                            rstride=rstride, cstride=cstride)

            if xlabel == None:
                ax.set_xlabel(
                    f'CV1 - {self.hills.cv_name[cv1_index]}', size=label_size)
            else:
                ax.set_xlabel(xlabel, size=label_size)
            if ylabel == None:
                ax.set_ylabel(
                    f'CV2 - {self.hills.cv_name[cv2_index]}', size=label_size)
            else:
                ax.set_ylabel(ylabel, size=label_size)
            if zlabel == None:
                # type: ignore
                ax.set_zlabel(f'free energy ({energy_unit})', size=label_size)
            else:
                ax.set_zlabel(zlabel, size=label_size)  # type: ignore
        else:
            raise ValueError(
                f"Surface plot only works for FES with exactly two CVs, and this FES has {self.hills.cvs}"
            )

    def removeCV(
        self,
        CV: int,
        kb: Optional[float] = None,
        energy_unit: str = "kJ/mol",
        temp: float = 300.0
    ):
        """Remove a CV from an existing FES. 
        The function first recalculates the FES to an array of probabilities. 
        The probabilities are summed along the CV to be removed, 
        and resulting probability distribution with 1 less dimension 
        is converted back to FES. 

        Interactivity was working in jupyter notebook/lab with "%matplotlib widget".

        Args:
            CV (int): the index of CV to be removed. 
            energy_unit (str): has to be either "kJ/mol" or "kcal/mol". Defaults to be "kJ/mol".
            kb (float, optional): the Boltzmann Constant in the energy unit. \
                Defaults to be None, which will be set according to energy_unit.
            temp (float) = temperature of the simulation in Kelvins.

        Return:
            New `Fes` instance without the CV to be removed.
        """
        
        print(f"Removing CV {CV}.")

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

        if CV > self.hills.cvs:
            print("Error: The CV to remove is not available in this FES object.")
            return None
        if kb == None:
            if energy_unit == "kJ/mol":
                kb = 8.314e-3
            elif energy_unit == "kcal/mol":
                kb = 8.314e-3 / 4.184
            else:
                raise ValueError(
                    "Please give the Boltzmann Constant in the energy unit.")

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

    def make_gif(
        self,
        gif_name: str = "FES.gif",
        cmap: str = "jet",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        label_size: int = 12,
        image_size: List[int] = [10, 7],
        opacity: float = 0.2,
        levels: Optional[List[float]] = None,
        frames: int = 64,
    ):
        """
        Generates an animation of the 3D free energy surface (FES) showing different isosurfaces.

        Usage:
        ```python
        fes.make_gif()
        ```

        Args:
            gif_name (str, default="FES.gif"): \
                The name of the gif file of the FES that will be saved in the current working directory.
            cmap (str, default="jet"): \
                The Matplotlib colormap used to color the 3D FES.
            xlabel, ylabel, zlabel (str, optional): \
                If provided, they will be used as labels for the graph.
            label_size (int, default=12): \
                The size of text in the labels.
            image_size (List[int], default=[10,7]): \
                The width and height of the picture.
            opacity (float, default=0.2): \
                A number between 0 and 1 representing the opacity of isosurfaces in the 3D FES.
            levels (List[float], optional): \
                A list of free energy values for isosurfaces in the 3D FES. If not provided, default values from the contours parameters will be used instead.
            frames (int, default=64): \
                The number of frames the animation will be composed of.
        """
        try:
            import pyvista as pv
        except (ImportError, ModuleNotFoundError) as e:
            print(e)
            sys.exit(1)

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

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
            raise ValueError(
                "Error: gif_plot is only available for FES with 3 CVs.")
