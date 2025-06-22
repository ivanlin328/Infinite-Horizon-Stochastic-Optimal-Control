import numpy as np
import copy

class ValueFunction:
    """
    Base class for a tabular value function over a discretized error state:
      error = [ex, ey, etheta],  theta is circular.
    """
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = np.array(ex_space)
        self.ey_space = np.array(ey_space)
        self.etheta_space = np.array(etheta_space)
        # self.v will be created in subclasses

    def copy_from(self, other):
        """
        Update this value function’s storage from another ValueFunction.
        """
        if not isinstance(other, ValueFunction):
            raise TypeError("Can only copy from another ValueFunction")
        self.__dict__ = copy.deepcopy(other.__dict__)

    def update(self, t, ex, ey, etheta, target_value):
        """
        Set V[t, ix, iy, ith] = target_value for the grid‐point closest to (ex, ey, etheta).
        """
        ix = int(np.argmin(np.abs(self.ex_space - ex)))
        iy = int(np.argmin(np.abs(self.ey_space - ey)))
        # handle circular wrap for theta
        diffs = np.arctan2(
            np.sin(etheta - self.etheta_space),
            np.cos(etheta - self.etheta_space)
        )
        ith = int(np.argmin(np.abs(diffs)))
        self.v[t, ix, iy, ith] = target_value

    def __call__(self, t, ex, ey, etheta):
        """
        Return V[t,ix,iy,ith] for the grid‐point closest to (ex,ey,etheta).
        """
        ix = int(np.argmin(np.abs(self.ex_space - ex)))
        iy = int(np.argmin(np.abs(self.ey_space - ey)))
        diffs = np.arctan2(
            np.sin(etheta - self.etheta_space),
            np.cos(etheta - self.etheta_space)
        )
        ith = int(np.argmin(np.abs(diffs)))
        return self.v[t, ix, iy, ith]

    def copy(self):
        """
        Return a deep copy of this ValueFunction.
        """
        new = self.__class__(
            self.T,
            self.ex_space.copy(),
            self.ey_space.copy(),
            self.etheta_space.copy()
        )
        new.v = self.v.copy()
        return new


class GridValueFunction(ValueFunction):
    """
    Tabular (grid) implementation storing V as a 4D array:
      V.shape = (T, len(ex_space), len(ey_space), len(etheta_space))
    """
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.v = np.zeros((
            T,
            len(self.ex_space),
            len(self.ey_space),
            len(self.etheta_space)
        ))


# class FeatureValueFunction(ValueFunction):
#     """
#     Feature-based value function
#     """
#     # TODO: your implementation
#     raise NotImplementedError


