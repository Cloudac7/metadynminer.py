import sys
import numpy as np
import pandas as pd

from itertools import product
from typing import Optional, List
from metadynminer.fes import Fes


class Minima:
    """
    Represents an object of the Minima class used to find local free energy minima on a free energy surface (FES).

    The FES is divided into a specified number of bins (default is 8), and the absolute minima is found for each bin. 
    The algorithm then checks if each found point is a local minimum by comparing it to the surrounding points on the FES.

    The list of minima is stored as a pandas DataFrame.

    Usage:
    ```python
    minima = metadynminer.Minima(fes=f, nbins=8)
    ```

    The list of minima can be accessed using the `minima.minima` attribute:
    ```python
    print(minima.minima)
    ```

    Args:
        fes (Fes): The Fes object to find the minima on.
        nbins (int, default=8): The number of bins used to divide the FES.
    """

    def __init__(
        self, fes: Fes, nbins=8
    ):
        if fes.fes is None:
            raise ValueError("FES is not defined.")
        self.fes = fes.fes
        self.periodic = fes.periodic
        self.cv_select = fes.cv_select
        self.cvs = len(self.cv_select)
        self.res = fes.res

        self.cv_name = [fes.hills.cv_name[i] for i in self.cv_select]

        # use remapped cv_min and cv_max
        self.cv_min = fes.cv_min
        self.cv_max = fes.cv_max

        self.cv_per = fes.hills.cv_per

        self.findminima(nbins=nbins)

    def findminima(self, nbins=8):
        """Method for finding local minima on FES.

        Args:
            fes (Fes): The Fes object to find the minima on.
            nbins (int, default=8): The number of bins used to divide the FES.
        """
        cv_min = self.cv_min
        cv_max = self.cv_max

        if int(nbins) != nbins:
            nbins = int(nbins)
            print(
                f"Number of bins must be an integer, it will be set to {nbins}.")
        if self.res % nbins != 0:
            print("Error: Resolution of FES must be divisible by number of bins.")
            return None
        if nbins > self.res/2:
            print("Error: Number of bins is too high.")
            return None
        bin_size = int(self.res/nbins)

        self.minima = None

        for index in np.ndindex(tuple([nbins] * self.cvs)):
            # index serve as bin number
            _fes_slice = tuple(
                slice(
                    index[i] * bin_size, (index[i] + 1) * bin_size
                ) for i in range(self.cvs)
            )
            fes_slice = self.fes[_fes_slice]
            bin_min = np.min(fes_slice)

            # indexes of global minimum of a bin
            bin_min_arg = np.unravel_index(
                np.argmin(fes_slice), fes_slice.shape
            )
            # indexes of that minima in the original fes (indexes +1)
            min_cv_b = np.array([
                bin_min_arg[i] + index[i] * bin_size for i in range(self.cvs)
            ], dtype=int)

            if (np.array(bin_min_arg, dtype=int) > 0).all() and \
                    (np.array(bin_min_arg, dtype=int) < bin_size - 1).all():
                # if the minima is not on the edge of the bin
                min_cv = (((min_cv_b+0.5)/self.res) * (cv_max-cv_min))+cv_min
                local_minima = np.concatenate([
                    [np.round(bin_min, 6)], min_cv_b, np.round(min_cv, 6)
                ])
                if self.minima is None:
                    self.minima = local_minima
                else:
                    self.minima = np.vstack((self.minima, local_minima))
            else:
                # if the minima is on the edge of the bin
                around = np.zeros(tuple([3] * self.cvs))

                for product_index in product(*[range(3)] * self.cvs):
                    converted_index = np.array(product_index, dtype=int) + \
                        np.array(min_cv_b, dtype=int) - 1
                    converted_index[self.periodic] = \
                        converted_index[self.periodic] % self.res

                    mask = np.ones(self.cvs, dtype=float)
                    mask[~self.periodic] = np.inf
                    fes_mask = np.prod(mask)

                    if product_index == tuple([1] * self.cvs):
                        around[product_index] = np.inf
                    else:
                        around[product_index] = \
                            self.fes[tuple(converted_index)] * fes_mask
                if bin_min < around.all():
                    min_cv = (((min_cv_b+0.5)/self.res)
                              * (cv_max-cv_min))+cv_min
                    local_minima = np.concatenate([
                        [np.round(bin_min, 6)], min_cv_b, np.round(min_cv, 6)
                    ])
                    if self.minima is None:
                        self.minima = local_minima
                    else:
                        self.minima = np.vstack((self.minima, local_minima))

        if self.minima is None:
            print("No minima found.")
            return None

        if len(self.minima.shape) > 1:
            self.minima = self.minima[self.minima[:, 0].argsort()]

        if self.minima.shape[0] == 1:
            self.minima = np.concatenate((
                np.arange(0, self.minima.shape[0], dtype=int), self.minima
            ))
        else:
            self.minima = np.column_stack((
                np.arange(0, self.minima.shape[0], dtype=int), self.minima
            ))

        minima_df = pd.DataFrame(
            np.array(self.minima),
            columns=["Minimum", "free energy"] +
            [f"CV{i+1}bin" for i in range(self.cvs)] +
            [f"CV{i+1} - {self.cv_name[i]}" for i in range(self.cvs)]
        )
        minima_df["Minimum"] = minima_df["Minimum"].astype(int)
        self.minima = minima_df

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
        color: Optional[str] = None,
        vmin: float = 0,
        vmax: Optional[float] = None,
        opacity: float = 0.2,
        levels: Optional[List[float]] = None,
        show_points: bool = True,
        point_size: float = 4.0,
    ):
        """
        Function used to visualize the FES objects with the positions of local minima shown as letters on the graph.

        Usage:
        ```python
        minima.plot()
        ```

        Args:
            png_name (str, optional): \
                If provided, the picture of the FES will be saved under this name to the current working directory.
            contours (bool, default=True): \
                Specifies whether contours should be shown on the 2D FES.
            contours_spacing (float, default=0.0): \
                When a positive number is set, it will be used as the spacing for contours on the 2D FES. \
                Otherwise, if contours=True, there will be five equally spaced contour levels.
            aspect (float, default=1.0): \
                The aspect ratio of the graph. Works with 1D and 2D FES.
            cmap (str, default="jet"): \
                The Matplotlib colormap used to color the 2D or 3D FES.
            energy_unit (str, default="kJ/mol"): \
                A string used in the description of the colorbar.
            xlabel, ylabel, zlabel (str, optional): \
                If provided, they will be used as labels for the graphs.
            label_size (int, default=12): The size of text in labels.
            image_size (List[int], default=[10,7]): The width and height of the picture.
            color (str, optional): \
                The name of the color in Matplotlib. \
                If set, the color will be used for the letters. \
                If not set, the color should be automatically either black or white, \
                depending on what will be more visible on the given place on the FES with the given colormap (for 2D FES).
            vmin (float, default=0): The lower bound for the colormap on the 2D FES.
            vmax (float, optional): The upper bound for the colormap on the 2D FES.
            opacity (float, default=0.2): \
                A number between 0 and 1 representing the opacity of isosurfaces in the 3D FES.
            levels (List[float], optional): \
                A list of free energy values for isosurfaces on the 3D FES. \
                If not provided, default values from the contours parameters will be used instead.
            show_points (bool, default=True): \
                Specifies whether points should be visualized instead of just the letters. \
                Only applicable for 3D FES.
            point_size (float, default=4.0): \
                The size of points if show_points=True.
        """

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if vmax == None:
            # if the addition is smaller than 0.01, the 3d plot stops working.
            vmax = np.max(self.fes) + 0.01

        if contours_spacing == 0.0:
            contours_spacing = (vmax-vmin)/5.0

        color_map = cm.get_cmap(cmap)

        color_map.set_over("white")
        color_map.set_under("white")

        color_set = True
        if color == None:
            color_set = False

        if self.cvs == 1:
            cv1min = self.cv_min[0]
            cv1max = self.cv_max[0]
            plt.figure(figsize=(image_size[0], image_size[1]))
            X = np.linspace(cv1min, cv1max, self.res)
            plt.plot(X, self.fes)

            if not color_set:
                color = "black"

            ferange = np.max(self.fes) - np.min(self.fes)

            if self.minima.shape[0] == 1:
                plt.text(float(self.minima.iloc[0, 3]), float(self.minima.iloc[0, 1])+ferange*0.05, self.minima.iloc[0, 0],
                         fontsize=label_size, horizontalalignment='center',
                         verticalalignment='bottom', c=color)
            elif self.minima.shape[0] > 1:
                for m in range(len(self.minima.iloc[:, 0])):
                    plt.text(float(self.minima.iloc[m, 3]), float(self.minima.iloc[m, 1])+ferange*0.05, self.minima.iloc[m, 0],
                             fontsize=label_size, horizontalalignment='center',
                             verticalalignment='bottom', c=color)

            if xlabel == None:
                plt.xlabel(f'CV1 - {self.cv_name[0]}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(f'free energy ({energy_unit})', size=label_size)
            else:
                plt.ylabel(ylabel, size=label_size)

        elif self.cvs == 2:
            cv1min, cv2min = self.cv_min[0], self.cv_min[1]
            cv1max, cv2max = self.cv_max[0], self.cv_max[1]
            fig = plt.figure(figsize=(image_size[0], image_size[1]))
            plt.imshow(np.rot90(self.fes, axes=(0, 1)), cmap=color_map, interpolation='nearest',
                       extent=[cv1min, cv1max, cv2min, cv2max],
                       aspect=(((cv1max-cv1min)/(cv2max-cv2min))/(aspect)),
                       vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.set_label(energy_unit, size=label_size)

            if self.minima.shape[0] == 1:
                background = color_map(
                    (float(self.minima.iloc[1])-vmin)/(vmax-vmin))
                luma = background[0]*0.2126+background[1] * \
                    0.7152+background[3]*0.0722
                if luma > 0.6 and not color_set:
                    color = "black"
                elif luma <= 0.6 and not color_set:
                    color = "white"
                plt.text(float(self.minima.iloc[0, 4]), float(self.minima.iloc[0, 5]), self.minima.iloc[0, 0],
                         fontsize=label_size, horizontalalignment='center',
                         verticalalignment='center', c=color)
            elif self.minima.shape[0] > 1:
                for m in range(len(self.minima.iloc[:, 0])):
                    background = color_map(
                        (float(self.minima.iloc[m, 1])-vmin)/(vmax-vmin))
                    luma = background[0]*0.2126 + \
                        background[1]*0.7152+background[3]*0.0722
                    if luma > 0.6 and not color_set:
                        color = "black"
                    elif luma <= 0.6 and not color_set:
                        color = "white"
                    plt.text(float(self.minima.iloc[m, 4]), float(self.minima.iloc[m, 5]), self.minima.iloc[m, 0],
                             fontsize=label_size, horizontalalignment='center',
                             verticalalignment='center', c=color)

            if contours:
                cont = plt.contour(np.rot90(self.fes, axes=(0, 1)),
                                   levels=np.arange(
                                       0, (vmax + 0.01), contours_spacing),
                                   extent=[cv1min, cv1max, cv2max, cv2min],
                                   colors="k")
                plt.clabel(cont, levels=np.arange(
                    0, (vmax + 0.01), contours_spacing))
            if xlabel == None:
                plt.xlabel(f'CV1 - {self.cv_name[0]}', size=label_size)
            else:
                plt.xlabel(xlabel, size=label_size)
            if ylabel == None:
                plt.ylabel(f'CV2 - {self.cv_name[1]}', size=label_size)
            else:
                plt.ylabel(ylabel, size=label_size)

        elif self.cvs == 3:
            try:
                import pyvista as pv
            except (ImportError, ModuleNotFoundError) as e:
                print(e)
                sys.exit(1)

            cv1min, cv2min, cv3min = self.cv_min[0], self.cv_min[1], self.cv_min[2]
            cv1max, cv2max, cv3max = self.cv_max[0], self.cv_max[1], self.cv_max[2]
            if xlabel == None:
                xlabel = "CV1 - " + self.cv_name[0]
            if ylabel == None:
                ylabel = "CV2 - " + self.cv_name[1]
            if zlabel == None:
                zlabel = "CV3 - " + self.cv_name[2]

            min_ar = self.minima.iloc[:, 5:8].values
            min_ar = min_ar.astype(np.float32)
            min_pv = pv.PolyData(min_ar)
            grid = pv.UniformGrid(
                dimensions=(self.res, self.res, self.res),
                spacing=((cv1max-cv1min)/self.res, (cv2max-cv2min) /
                         self.res, (cv3max-cv3min)/self.res),
                origin=(cv1min, cv2min, cv3min)
            )
            grid["vol"] = self.fes.ravel(order="F")
            if levels == None:
                contours = grid.contour(
                    np.arange(0, (vmax - 0.1), contours_spacing))
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
                       cmap=color_map, show_scalar_bar=False, interpolate_before_map=True)
            p.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            p.add_point_labels(min_pv, self.minima.iloc[:, 0],
                               show_points=show_points, always_visible=True,
                               point_color="black", point_size=point_size,
                               font_size=label_size, shape=None)
            p.show()

        if png_name != None:
            plt.savefig(png_name)

    def make_gif(self, gif_name="FES.gif", cmap="jet",
                 xlabel=None, ylabel=None, zlabel=None, label_size=12, image_size=[10, 7],
                 opacity=0.2, levels=None, show_points=True, point_size=4.0, frames=64):
        """
        Equvivalent to Fes.make_gif()
        """

        try:
            import pyvista as pv
        except (ImportError, ModuleNotFoundError) as e:
            print(e)
            sys.exit(1)

        if self.cvs == 3:
            cv1min, cv2min, cv3min = self.cv_min[0], self.cv_min[1], self.cv_min[2]
            cv1max, cv2max, cv3max = self.cv_max[0], self.cv_max[1], self.cv_max[2]
            values = np.linspace(np.min(self.fes)+1,
                                 np.max(self.fes), num=frames)
            grid = pv.UniformGrid(
                dimensions=(self.res, self.res, self.res),
                spacing=((cv1max-cv1min)/self.res, (cv2max-cv2min) /
                         self.res, (cv3max-cv3min)/self.res),
                origin=(cv1min, cv2min, cv3min),
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
                plotter.show_grid(
                    xlabel=f"CV1 - {self.cv_name[0]}", ylabel=f"CV2 - {self.cv_name[1]}", zlabel=f"CV3 - {self.cv_name[2]}")
            else:
                plotter.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            if show_points:
                min_ar = self.minima.iloc[:, 5:8].values
                min_ar = min_ar.astype(np.float32)
                min_pv = pv.PolyData(min_ar)
                plotter.add_point_labels(min_pv, self.minima.iloc[:, 0],
                                         show_points=True, always_visible=True,
                                         pickable=True, point_color="black",
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
