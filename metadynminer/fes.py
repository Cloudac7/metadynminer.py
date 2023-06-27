import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm, trange
from typing import Optional, List, Union, Dict, Any
from matplotlib.colors import Colormap
from metadynminer.hills import Hills

from miko.graph.plotting import canvas_style, AxesInit, square_grid
from miko.utils.log_factory import logger


class FES:
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
        time_min (int): The starting time step of simulation. Defaults to 0.
        time_max (int, optional): The ending time step of simulation. Defaults to None.
    """

    def __init__(
        self,
        hills: Hills,
        original: bool = False,
        calculate_new_fes: bool = True,
        resolution: int = 256,
        time_min: int = 0,
        time_max: Optional[int] = None
    ):
        self.res = resolution
        self.fes = None
        self.cvs = hills.cvs

        self.hills = hills
        self.periodic = hills.get_periodic()
        self.cv_name = hills.cv_name

        if time_max != None:
            if time_max <= time_min:
                logger.warning("Warning: End time is lower than start time")
            if time_max > len(self.hills.cv[:, 0]):
                time_max = len(self.hills.cv[:, 0])
                logger.warning(
                    f"Warning: End time {time_max} is higher than",
                    f"number of lines in HILLS file {len(self.hills.cv[:, 0])},",
                    "which will be used instead."
                )

        if calculate_new_fes:
            if not original:
                self.makefes(resolution, time_min, time_max)
            else:
                self.makefes2(resolution, time_min, time_max)

    def generate_cv_map(self):
        """generate CV map"""

        cv_min = self.hills.cv_min
        cv_max = self.hills.cv_max
        cv_range = cv_max - cv_min

        self.cv_range = cv_range

        cv_min[~self.periodic] -= cv_range[~self.periodic] * 0.15
        cv_max[~self.periodic] += cv_range[~self.periodic] * 0.15
        cv_fes_range = np.abs(cv_max - cv_min)

        # generate remapped cv_min and cv_max
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.cv_fes_range = cv_fes_range

    def makefes(
        self,
        resolution: Optional[int] = None,
        time_min: Optional[int] = None,
        time_max: Optional[int] = None
    ):
        """Function used internally for summing hills in Hills object with the fast Bias Sum Algorithm. 

        Args:
            resolution (int, optional): \
                The resolution of the free energy surface. Defaults to 256.
            time_min (int): The starting time step of simulation. Defaults to 0.
            time_max (int, optional): The ending time step of simulation. Defaults to None.
        """

        if resolution is None:
            resolution = self.res
        self.generate_cv_map()

        cv_min = self.cv_min
        cv_max = self.cv_max
        cv_fes_range = self.cv_fes_range

        cv_bins = np.array([np.ceil(
            (self.hills.cv[time_min:time_max, cv_index] -
             cv_min[cv_index]) * resolution / cv_fes_range[cv_index]
        ) for cv_index in range(self.cvs)])
        cvs = self.cvs
        cv_bins = cv_bins.astype(int)

        sigma = np.array([self.hills.sigma[cv_index][0]
                         for cv_index in range(self.cvs)])
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
                        (fes_index_to_edit[d] < 0) + (
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
        time_min: Optional[int] = None,
        time_max: Optional[int] = None
    ):
        """
        Function internally used to sum Hills in the same way as Plumed `sum_hills`. 

        Args:
            resolution (int, optional): \
                The resolution of the free energy surface. Defaults to 256.
            time_min (int): The starting time step of simulation. Defaults to 0.
            time_max (int, optional): The ending time step of simulation. Defaults to None.
        """

        if resolution is None:
            resolution = self.res

        self.generate_cv_map()
        cv_min = self.cv_min
        cv_max = self.cv_max
        cv_fes_range = self.cv_fes_range

        cvs = self.cvs
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
            for i, cv_idx in enumerate(range(cvs)):
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

    def remove_cv(
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
            New `FES` instance without the CV to be removed.
        """

        logger.info(f"Removing CV {CV}.")

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

        if CV > self.hills.cvs:
            raise ValueError("Error: The CV to remove is not available in this FES object.")

        if kb == None:
            if energy_unit == "kJ/mol":
                kb = 8.314e-3
            elif energy_unit == "kcal/mol":
                kb = 8.314e-3 / 4.184
            else:
                raise ValueError(
                    "Please give the Boltzmann Constant in the energy unit.")

        if self.cvs == 1:
            raise ValueError("Error: You can not remove the only CV. ")
        else:
            probabilities = np.exp(-self.fes / (kb * temp))
            new_prob = np.sum(probabilities, axis=CV)

            new_fes = FES(hills=self.hills, calculate_new_fes=False)
            new_fes.fes = - kb * temp * np.log(new_prob)
            new_fes.fes = new_fes.fes - np.min(new_fes.fes)
            new_fes.res = self.res

            mask = np.ones(self.cvs, dtype=bool)
            mask[CV] = False
            new_fes.cv_min = self.cv_min[mask]
            new_fes.cv_max = self.cv_max[mask]
            new_fes.cv_fes_range = self.cv_fes_range[mask]
            new_fes.cv_name = [
                j for i, j in enumerate(self.cv_name) if mask[i]]
            new_fes.cvs = self.cvs - 1
            return new_fes

    def remove_cvs(
        self,
        CVs: List[int],
        kb: Optional[float] = None,
        energy_unit: str = "kJ/mol",
        temp: float = 300.0
    ):
        fes = self.remove_cv(CVs[0], kb, energy_unit, temp)
        if len(CVs) > 1:
            for CV in CVs[1:]:
                if fes is not None:
                    fes = fes.remove_cv(CV, kb, energy_unit, temp)
        return fes

    def set_fes(self, fes: np.ndarray):
        self.fes = fes

    def plot(
        self,
        png_name: Optional[str] = None,
        cmap: Union[str, Colormap] = "RdYlBu_r",
        energy_unit: str = "kJ/mol",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        image_size: List[int] = [10, 7],
        levels: Optional[int] = 20,
        dpi: int = 96,
        surface: bool = False,
        surface_params: Optional[Dict[str, Any]] = None,
        **kwargs
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
            cmap (str, default="RdYlBu"): The Matplotlib colormap used to color the 2D or 3D FES.
            energy_unit (str, default="kJ/mol"): The unit used in the description of the colorbar.
            xlabel, ylabel, zlabel (str, optional): If provided, they will be used as labels for the graphs.
            label_size (int, default=12): The size of text in the labels.
            image_size (List[int], default=[10,7]): The width and height of the picture.
            vmin (float, default=0): The lower bound for the colormap on the 2D FES.
            vmax (float, optional): The upper bound for the colormap on the 2D FES.
            opacity (float, default=0.2): A number between 0 and 1 that represents the opacity of isosurfaces in the 3D FES.
            levels (int, optional): A list of free energy values for isosurfaces in FES. Defaults to be 20.
        """

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

        if type(cmap) is str:
            cmap = cm.get_cmap(cmap)

        cvs = self.cvs

        if cvs == 1:
            fig, ax = PlottingFES._plot1d(
                self, image_size=image_size, dpi=dpi,
                energy_unit=energy_unit, xlabel=xlabel, **kwargs
            )

        elif cvs == 2:
            if surface:
                fig, ax = PlottingFES._surface_plot(
                    self, 
                    cmap=cmap, image_size=image_size, dpi=dpi,
                    xlabel=xlabel, ylabel=ylabel,
                    energy_unit=energy_unit,
                    **surface_params, **kwargs
                )
            fig, ax = PlottingFES._plot2d(
                self, 
                levels=levels, cmap=cmap, image_size=image_size, dpi=dpi,
                xlabel=xlabel, ylabel=ylabel, **kwargs
            )

        else:
            raise ValueError("Only 1D and 2D FES are supported.")

        if png_name != None:
            fig.savefig(png_name)

        return fig, ax


class PlottingFES:
    """Plot FES from a FES object.
    """

    @staticmethod
    def _surface_plot(
        fes: FES,
        cmap: Union[str, Colormap] = "RdYlBu",
        energy_unit: str = "kJ/mol",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        label_size: int = 12,
        image_size: List[int] = [10, 7],
        rstride: int = 1,
        cstride: int = 1,
        dpi: int = 96,
        **kwargs
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

        cv_select = list(range(fes.cvs))

        if fes.cvs == 2:
            cv1min = fes.cv_min[fes.cv_select[0]]
            cv1max = fes.cv_max[fes.cv_select[0]]
            cv2min = fes.cv_min[fes.cv_select[1]]
            cv2max = fes.cv_max[fes.cv_select[1]]

            x = np.linspace(cv1min, cv1max, fes.res)
            y = np.linspace(cv2min, cv2max, fes.res)

            X, Y = np.meshgrid(x, y)
            Z = fes.fes

            canvas_style(**kwargs)

            fig, ax = plt.subplots(
                figsize=image_size, dpi=dpi,
                subplot_kw={"projection": "3d"}
            )
            ax.plot_surface(X, Y, Z, cmap=cmap,  # type: ignore
                            rstride=rstride, cstride=cstride)

            if xlabel == None:
                ax.set_xlabel(
                    f'CV1 - {fes.cv_name[0]}', size=label_size)
            else:
                ax.set_xlabel(xlabel, size=label_size)
            if ylabel == None:
                ax.set_ylabel(
                    f'CV2 - {fes.cv_name[1]}', size=label_size)
            else:
                ax.set_ylabel(ylabel, size=label_size)
            if zlabel == None:
                # type: ignore
                ax.set_zlabel(f'Free energy ({energy_unit})', size=label_size)
            else:
                ax.set_zlabel(zlabel, size=label_size)  # type: ignore
        else:
            raise ValueError(
                f"Surface plot only works for FES with exactly two CVs, and this FES has {fes.hills.cvs}"
            )
        return fig, ax

    @staticmethod
    def _plot1d(
        fes: FES,
        image_size: List[int] = [10, 7],
        dpi: int = 96,
        energy_unit: str = 'kJ/mol',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        **kwargs
    ):
        canvas_style(**kwargs)
        fig, ax = plt.subplots(
            figsize=(image_size[0], image_size[1]),
            dpi=dpi
        )
        X = np.linspace(fes.cv_min[0], fes.cv_max[0], fes.res)
        ax.plot(X, fes.fes)
        if xlabel == None:
            ax.set_xlabel(
                f'CV1 - {fes.cv_name[0]}')
        else:
            ax.set_xlabel(xlabel)
        if ylabel == None:
            ax.set_ylabel(f'Free Energy ({energy_unit})')
        else:
            ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def _plot2d(
        fes_obj: FES,
        levels: int = 20,
        cmap: Union[str, Colormap] = "RdYlBu",
        image_size: List[int] = [10, 7],
        dpi: int = 96,
        energy_unit: str = 'kJ/mol',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        **kwargs
    ):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        canvas_style(**kwargs)
        fig, ax = plt.subplots(
            figsize=(image_size[0], image_size[1]),
            dpi=dpi
        )

        if fes_obj.fes is None:
            raise ValueError(
                "FES not calculated yet. Use fes.makefes() or fes.makefes2() first.")
        X = np.linspace(fes_obj.cv_min[0], fes_obj.cv_max[0], fes_obj.res)
        Y = np.linspace(fes_obj.cv_min[1], fes_obj.cv_max[1], fes_obj.res)
        fes = fes_obj.fes.T
        cs = ax.contourf(X, Y, fes, levels=levels, cmap=cmap)
        ax.contour(X, Y, fes, levels=levels, colors="black", alpha=0.2)

        if xlabel == None:
            ax.set_xlabel(
                f'CV1 - {fes_obj.cv_name[0]}')
        else:
            ax.set_xlabel(xlabel)
        if ylabel == None:
            ax.set_ylabel(
                f'CV2 - {fes_obj.cv_name[1]}')
        else:
            ax.set_ylabel(ylabel)
        cbar = fig.colorbar(cs)
        if zlabel == None:
            cbar.set_label(f'Free Energy ({energy_unit})')
        else:
            cbar.set_label(zlabel)
        return fig, ax
