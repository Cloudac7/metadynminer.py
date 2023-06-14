import sys
import numpy as np
import pandas as pd

from itertools import product
from metadynminer.fes import Fes


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

    - fes = Fes object to find the minima on
    - nbins (default = 8) = number of bins to divide the FES
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
        """
        Internal method for finding local minima on FES.
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

    def plot(self, png_name=None, contours=True, contours_spacing=0.0, aspect=1.0, cmap="jet",
             energy_unit="kJ/mol", xlabel=None, ylabel=None, zlabel=None, label_size=12, image_size=[10, 7],
             color=None, vmin=0, vmax=None, opacity=0.2, levels=None, show_points=True, point_size=4.0):
        """
        The same function as for visualizing Fes objects, but this time 
        with the positions of local minima shown as letters on the graph.

        ```python
        minima.plot()
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
        - color = string = name of color in matplotlib, if set, the color will be used for the letters. 
                If not set, the color should be automatically either black or white, 
                depending on what will be better visible on given place on FES with given colormap (for 2D FES).
        - vmin (default=0) = real number, lower bound for the colormap on 2D FES
        - vmax = real number, upper bound for the colormap on 2D FES
        - opacity (default=0.2) = number between 0 and 1, is the opacity of isosurfaces of 3D FES
        - levels = Here you can specify list of free energy values for isosurfaces on 3D FES. 
                If not provided, default values from contours parameters will be used instead. 
        - show_points (default=True) = boolean, tells if points should be visualized too, instead of just the letters. Only on 3D FES. 
        - point_size (default=4.0) = float, sets the size of points if show_points=True
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if vmax == None:
            # if the addition is smaller than 0.01, the 3d plot stops working.
            vmax = np.max(self.fes)+0.01

        if contours_spacing == 0.0:
            contours_spacing = (vmax-vmin)/5.0

        cmap = cm.get_cmap(cmap)

        cmap.set_over("white")
        cmap.set_under("white")

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
            plt.imshow(np.rot90(self.fes, axes=(0, 1)), cmap=cmap, interpolation='nearest',
                       extent=[cv1min, cv1max, cv2min, cv2max],
                       aspect=(((cv1max-cv1min)/(cv2max-cv2min))/(aspect)),
                       vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.set_label(energy_unit, size=label_size)

            if self.minima.shape[0] == 1:
                background = cmap(
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
                    background = cmap(
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
                       cmap=cmap, show_scalar_bar=False, interpolate_before_map=True)
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
