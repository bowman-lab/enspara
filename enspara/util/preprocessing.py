import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing.data import _handle_zeros_in_scale

from enspara import exception

class ResidueTypeScaler(BaseEstimator, TransformerMixin):
    """Similar to StandardScaler(with_mean=False) but
    aggregates across all columns representing the same
    residue.

    Parameters
    ----------
    scale_func : callable
        Function used to compute the factor by which all columns will
        be scaled. Must accept an array of shape
        (n_observations, n_residues) and return a single value.
    top : md.Topology or similar
        Protein topology. Used to know columns' residue type.
    copy : bool, default=True
        When transform is run, should data be copied (rather than scaled
        in place.)
    """

    def __init__(self, scale_func, top, copy=True):
        self.scale_func = scale_func
        self.top = top
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the scaling factor for each residue type.

        Parameters
        ----------
        X : np.ndarray, shape=(n_observations, n_residues)
            Array of values to fit scaling upon.
        y : Passthrough for Pipeline compatibility.
        """

        if X.shape[1] != self.top.n_residues:
            raise exception.InvalidData("Given data had shape {s} and top had n_residues {n}".format(s=X.shape, n=self.top.n_residues))

        self.scale_factors_ = {}
        for code, residues in self.code2rindex.items():
            if code is None:
                warnings.warn(exception.SuspiciousDataWarning("ResidueTypeScaler Topology had 'None' values as residue codes. These will be scaled as though they are the same residue type."))

            target_data = X[:, residues]
            scale_factor = _handle_zeros_in_scale(self.scale_func(target_data), copy=False)

            self.scale_factors_[code] = scale_factor

        return self

    def transform(self, x, copy=None):
        """Perform standardization by scaling each residue type.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        assert hasattr(self, 'scale_factors_')

        copy = copy if copy is not None else self.copy
        if copy:
            x = x.copy()

        for code, residues in self.code2rindex.items():
            scale_factor = self.scale_factors_[code]
            x[:, residues] /= scale_factor

        return x

    @property
    def code2rindex(self):
        d = {}
        for i in range(self.top.n_residues):
            d.setdefault(self.top.residue(i).code, []).append(i)
        return d
