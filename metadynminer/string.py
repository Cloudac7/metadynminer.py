"""
From GitHub Repository: https://github.com/apallath/stringmethod
@author Akash Pallath
Licensed under the MIT license, see LICENSE for details.

Reference: Weinan E, "Simplified and improved string method for computing the minimum energy paths in barrier-crossing events",
J. Chem. Phys. 126, 164103 (2007), https://doi.org/10.1063/1.2720838
"""


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf, griddata, interp1d
from tqdm import tqdm, trange
from typing import Literal, List, Union, Tuple, Optional

from miko.utils import logger
from metadynminer.fes import FES
from metadynminer.minima import Minima


class String2D:
    """
    Class containing methods to compute the minimum energy path between two
    points on an energy landscape $V$.

    Args:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
            Missing values should be set to np.inf.
        indexing: Indexing of V array ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').

    Attributes:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
        X: Grid of shape (ny, nx) or (nx, ny) containing x-coordinates of each point on the grid.
        Y: Grid of shape (ny, nx) or (nx, ny) containing y-coordinates of each point on the grid.
        indexing: Indexing of V, X, and Y arrays ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').
        gradX: Gradient in x.
        gradY: Gradient in y.
        string_traj: Trajectory showing the evolution of the string (default=[]).
        mep: Converged minimum energy path (default=None, if not converged).
    """

    def __init__(self, fes: FES, indexing: Literal['xy', 'ij'] = 'xy'):
        self.fes = fes
        try:
            self.x = np.linspace(fes.cv_min[0], fes.cv_max[0], fes.res)
            self.y = np.linspace(fes.cv_min[1], fes.cv_max[1], fes.res)
        except IndexError:
            raise ValueError(
                "FES should contain at least 2 CVs."
            )
        if fes.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first."
            )
        else:
            self.V = fes.fes.T

        # Generate grids
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing=indexing)
        self.grid = np.vstack([self.X.ravel(), self.Y.ravel()]).T

        # Compute gradients
        self.indexing = indexing

        if self.indexing == 'xy':
            self.gradY, self.gradX = np.gradient(self.V, self.x, self.y)
        elif self.indexing == 'ij':
            self.gradX, self.gradY = np.gradient(self.V, self.x, self.y)
        else:
            raise ValueError("Indexing method not recognized.")

        # String method variables
        self.string_traj = []
        self.mep = None

    def compute_mep(
        self,
        begin: Union[np.ndarray, List[int], Tuple[int, int]],
        end: Union[np.ndarray, List[int], Tuple[int, int]],
        mid: List[Union[np.ndarray, List[int], Tuple[int, int]]] = [],
        function: str = 'linear',
        npts: int = 100,
        integrator: str = "forward_euler",
        dt: float = 0.1,
        tol: Optional[float] = None,
        maxsteps: int = 100,
        traj_every: int = 10,
        flexible: bool = True,
    ):
        """
        Computes the minimum free energy path. The points `begin`
        and `end` and the midpoints passed through `mid` are used to generate
        an initial guess (a k-order spline which interpolates through all the points).
        If no midpoints are defined, then the initial guess is a line connecting `begin`
        and `end`. If `flexible` is set to False, the ends of the string are fixed to `begin`
        and `end`, otherwise the ends of the string are free to move.

        Args:
            begin: Array of shape (2,) specifying starting point of the string.
            end: Array of shape (2,) specifying end point of the string.
            mid: List of arrays of shape (2,) specifying points between `begin` and `end`
                to use for generating an initial guess of the minimum energy path (default=[]).
            function: The radial basis function used for interpolation. (default='linear').
            npts: Number of points between any two valuesalong the string (default=100).
            integrator: Integration scheme to use (default='forward_euler'). Options=['forward_euler'].
            dt: Integration timestep (default=0.1).
            tol: Convergence criterion; stop stepping if string has an RMSD < tol between
                consecutive steps (default = max{npts^-4, 10^-10}).
            maxsteps: Maximum number of steps to take (default=100).
            traj_every: Interval to store string trajectory (default=10).
            flexible: If False, the ends of the string are fixed (default=True).

        Returns:
            mep: Array of shape (npts, 2) specifying string images along the minimum energy path between `begin` and `end`.
        """
        # Calculate params
        if tol is None:
            tol = max([npts**-4, 1e-10])

        # Generate initial guess
        if len(mid) > 0:
            string_x = np.linspace(begin[0], end[0], npts)
            xpts = [begin[0]] + [mpt[0] for mpt in mid] + [end[0]]
            ypts = [begin[1]] + [mpt[1] for mpt in mid] + [end[1]]
            spline = Rbf(xpts, ypts, function=function)
            string_y = spline(string_x)
        else:
            string_x = np.linspace(begin[0], end[0], npts)
            string_y = np.linspace(begin[1], end[1], npts)

        string = np.vstack([string_x, string_y]).T

        # Store initial guess
        self.string_traj = []
        self.string_traj.append(string)

        # Loop
        old_string = np.zeros_like(string)

        for tstep in trange(1, maxsteps + 1):
            # Integrator step
            if integrator == "forward_euler":
                old_string[:] = string
                string = self.step_euler(string, dt, flexible=flexible)
            else:
                raise ValueError("Invalid integrator")

            # Reparameterize string (equal arc length reparameterization)
            arclength = np.hstack(
                [0, np.cumsum(np.linalg.norm(string[1:] - string[:-1], axis=1))])
            arclength /= arclength[-1]
            reparam_x = interp1d(arclength, string[:, 0])
            reparam_y = interp1d(arclength, string[:, 1])
            gamma = np.linspace(0, 1, npts)
            string = np.vstack([reparam_x(gamma), reparam_y(gamma)]).T

            # Store
            string_change = np.sqrt(np.mean((string - old_string) ** 2))
            if tstep % traj_every == 0:
                self.string_traj.append(string)
                # Print convergence
                logger.info(
                    "Change in string: {:.10f}".format(string_change)
                )

            # Test for convergence
            if string_change < tol:
                logger.info("Change in string lower than tolerance.")
                logger.info(f"Converged in {tstep + 1} steps.")
                break

        # Store minimum energy path
        self.mep = string

    def load_minima(
        self,
        minima: Optional[Minima] = None,
        nbins: int = 8
    ):
        if minima is None:
            minima = Minima(self.fes, nbins)
        self.minima = minima.minima
        logger.info(self.minima)

    def mep_from_minima(
        self,
        begin_index: int,
        end_index: int,
        mid_indices: List[int] = [],
        *,
        minima: Optional[Minima] = None,
        nbins: int = 8,
        **kwargs
    ):
        """Calculate MEP from given reaction path through index of local minima.

        Args:
            begin_index (int): The index of starting local minima.
            end_index (int): The index of ending local minima.
            mid_indices (List[int], optional): The index of path in the middle of reaction path. Defaults to [].
            minima (Optional[Minima], optional): If minima not loaded, please select here. Defaults to None.
            nbins (int, optional): Refer to minima.Minima. Defaults to 8.

        Raises:
            ValueError: `self.minima` should not be None, or please provide one into `minima` to load it.
        """
        if self.minima is None:
            if minima is not None:
                self.load_minima(minima, nbins)
            else:
                raise ValueError(
                    "No minima found. Please run `load_minima` or pass a `Minima` object."
                )
        minima_x = self.minima.filter(regex=r"^CV1\s+-\s+") # type: ignore
        minima_y = self.minima.filter(regex=r"^CV2\s+-\s+") # type: ignore
        begin_x = minima_x.iloc[begin_index].values[0]
        begin_y = minima_y.iloc[begin_index].values[0]
        end_x = minima_x.iloc[end_index].values[0]
        end_y = minima_y.iloc[end_index].values[0]
        if len(mid_indices) > 0:
            mid = [[minima_x.iloc[i].values[0], minima_y.iloc[i].values[0]]
                   for i in mid_indices]
        else:
            mid = []
        self.compute_mep(
            begin=[begin_x, begin_y],
            mid=mid,
            end=[end_x, end_y],
            **kwargs
        )

    def step_euler(self, string, dt, flexible=True):
        """
        Evolves string images in time in response to forces calculated from the energy landscape.

        Args:
            string: Array of shape (npts, 2) specifying string images at the previous timestep.
            dt: Timestep.
            flexible: If False, the ends of the string are fixed (default=True).

        Returns:
            newstring: Array of shape (npts, 2) specifying string images after a timestep.
        """
        # Compute gradients at string points
        string_grad_x = griddata(
            self.grid, self.gradX.ravel(), string, method="linear")
        string_grad_y = griddata(
            self.grid, self.gradY.ravel(), string, method="linear")
        h = np.max(np.sqrt(string_grad_x**2 + string_grad_y**2))

        # Euler step
        if flexible:
            string = string - dt * \
                np.vstack([string_grad_x, string_grad_y]).T / h
        else:
            string[1:-1] = (
                string[1:-1] - dt *
                np.vstack([string_grad_x, string_grad_y]).T[1:-1] / h
            )

        return string

    def get_mep_energy_profile(self):
        energy_mep = griddata(self.grid, self.V.ravel(),
                              self.mep, method="linear")
        return self.mep, energy_mep

    def plot_V(self, clip_min=None, clip_max=None, levels=None, cmap="RdYlBu", dpi=300):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        fig, ax = plt.subplots(dpi=dpi)

        V = self.V
        if clip_min is not None:
            V = V.clip(min=clip_min)
        if clip_max is not None:
            V = V.clip(max=clip_max)

        cs = ax.contourf(self.X, self.Y, V, levels=levels, cmap=cmap)
        ax.contour(self.X, self.Y, V, levels=levels, colors="black", alpha=0.2)
        cbar = fig.colorbar(cs)
        return fig, ax, cbar

    def plot_mep(self, **plot_V_kwargs):
        """
        Plots the minimum energy path on the energy landscape $V$.

        Args:
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax, cbar = self.plot_V(**plot_V_kwargs)
        ax.scatter(self.mep[0, 0], self.mep[0, 1], color="white")
        ax.scatter(self.mep[-1, 0], self.mep[-1, 1], color="white")
        ax.plot(self.mep[:, 0], self.mep[:, 1], color="white")
        return fig, ax, cbar

    def plot_mep_energy_profile(self, dpi=300):
        """
        Plots the energy profile along the minimum energy path in $V$.
        """
        energy_mep = griddata(self.grid, self.V.ravel(),
                              self.mep, method="linear")
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(np.linspace(0, 1, len(energy_mep)), energy_mep)
        return fig, ax

    def plot_string_evolution(self, string_cmap=cm.gray, **plot_V_kwargs):
        """
        Plots the evolution of the string on the energy landscape $V$.

        Args:
            string_cmap: Colormap to use for plotting the evolution of the string.
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax, cbar = self.plot_V(**plot_V_kwargs)
        colors = string_cmap(np.linspace(0, 1, len(self.string_traj)))
        for sidx, string in enumerate(self.string_traj):
            ax.plot(string[:, 0], string[:, 1], "--", color=colors[sidx])
        return fig, ax, cbar
