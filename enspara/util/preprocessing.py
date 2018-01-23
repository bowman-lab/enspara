from sklearn.base import BaseEstimator, TransformerMixin


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

        assert X.shape[1] == self.top.n_residues

        self.scale_factors_ = {}
        for code, residues in self.code2rindex.items():
            target_data = X[:, residues]
            scale_factor = self.scale_func(target_data)

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
            if scale_factor != 0:
                x[:, residues] /= scale_factor

        return x

    @property
    def code2rindex(self):
        d = {}
        for i in range(self.top.n_residues):
            d.setdefault(self.top.residue(i).code, []).append(i)
        return d
