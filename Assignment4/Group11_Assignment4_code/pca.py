"""
pca.py — PCA dimension reduction for Assignment-4 Task-1.

Method
------
Covariance-eigenvector PCA, fitted on the TRAINING SPLIT ONLY.

A4 requires this precisely: "The directions of projection (eigen vectors) are
obtained from training data. The mean subtracted training, mean subtracted
validation, and mean subtracted test data are projected onto directions of
projection (eigen vectors)". So the eigenvectors and the mean both come from
train; val and test are only projected. Fitting on train+val+test would leak test
information into the representation and inflate every downstream accuracy.

    mean      mu      = (1/N) sum_i x_i                          (train only)
    centered  x~_i    = x_i - mu
    covariance C      = (1/(N-1)) sum_i x~_i x~_i^T              (784 x 784)
    eigen     C       = U L U^T      via eigh (symmetric, ascending)
    components W_k    = U[:, argsort(-L)][:, :k]                 (784 x k)
    reduced   z_i     = x~_i . W_k                               (k-dim)

Sample covariance uses 1/(N-1), not 1/N, so the eigenvalues are unbiased sample
variances. This does not change the eigenvectors, only their scale, but it makes
"variance retained" a meaningful number for the report.

Numerics — fp32 throughout, never fp16/autocast. The covariance matrix and its
eigendecomposition must be bit-reproducible: torch.linalg.eigh is deterministic
for a fixed input on a fixed device, and mixing precisions here would make the
reported variance-retained figures irreproducible.

variance_retained(k) = sum(L[:k]) / sum(L), the fraction of total variance the
top-k components preserve. Reported per dimension in Task-1 and used in the
report to explain why accuracy rises with dimension.
"""

import torch

from run_tracker import atomic_save


class PCA:
    """
    Fitted PCA on 784-d flattened images.

    Attributes:
        mean      (784,)   training-set mean vector
        components(784, k) top-k eigenvectors, columns, descending eigenvalue
        eigenvalues(784,)  full spectrum, descending
        k        int       retained dimension
    """

    def __init__(self, k=32):
        self.k = k
        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.total_variance = None

    def fit(self, X_train):
        """
        Compute mean and eigenvectors from X_train only.

        X_train: (N, 784) float32 on any device.
        Returns self.
        """
        # Keep the accumulation in fp32 explicitly. X is already fp32, but
        # pin the dtype so an accidental upstream cast cannot silently change
        # the covariance precision.
        X = X_train.to(torch.float32)
        self.mean = X.mean(dim=0)

        # Center in place on a copy so X_train is never mutated: the caller
        # reuses X_train for raw-input baselines and autoencoder training, and
        # an in-place subtract here would silently corrupt those.
        Xc = X - self.mean

        # Unbiased sample covariance. Symmetrize before eigh: the Gram matrix is
        # symmetric in exact arithmetic, but floating-point accumulation can leave
        # it minutely asymmetric, and eigh reads only one triangle. Symmetrizing
        # makes the input exactly symmetric and the result well-defined.
        N = Xc.size(0)
        C = (Xc.T @ Xc) / (N - 1)
        C = 0.5 * (C + C.T)

        # eigh returns ascending eigenvalues with eigenvectors as columns.
        # Run on the CPU: cuSOLVER's syevd is not guaranteed bitwise identical
        # across driver versions, and the whole decomposition is ~1-2 s on 784x784.
        # Determinism here matters more than the speed, since the reported
        # variance-retained figures must be reproducible.
        eigenvalues, eigenvectors = torch.linalg.eigh(C.cpu())

        # Descending order: largest variance first, so the first k columns are the
        # top-k principal directions.
        order = torch.argsort(eigenvalues, descending=True)
        self.eigenvalues = eigenvalues[order].contiguous()
        self.components = eigenvectors[:, order].contiguous()
        self.total_variance = self.eigenvalues.sum()

        # Back to the caller's device so projections run where the data lives.
        self.mean = self.mean.to(X.device)
        self.eigenvalues = self.eigenvalues.to(X.device)
        self.components = self.components.to(X.device)
        self.total_variance = self.total_variance.to(X.device)

        return self

    def transform(self, X, apply_mean=True):
        """
        Project X into the k-dim space: z = (x - mean) @ components.

        apply_mean=True subtracts the TRAINING mean, which is what A4 mandates for
        val and test. Pass False only to project already-centered data.
        """
        Z = X.to(torch.float32)
        if apply_mean:
            Z = Z - self.mean
        return Z @ self.components[:, :self.k]

    def variance_retained(self):
        """Fraction of total variance captured by the top-k components."""
        if self.eigenvalues is None:
            raise RuntimeError("PCA not fitted")
        return float(self.eigenvalues[:self.k].sum() / self.total_variance)

    def explained_variance_curve(self, points=None):
        """
        Cumulative variance retained at each dimension, for the report's figure.

        points: optional list of dimensions to sample (default: 1..784).
        Returns a tensor of cumulative fractions at those points.
        """
        if self.eigenvalues is None:
            raise RuntimeError("PCA not fitted")
        pts = points or list(range(1, self.eigenvalues.numel() + 1))
        cum = torch.cumsum(self.eigenvalues, dim=0) / self.total_variance
        return torch.stack([cum[p - 1] for p in pts])

    def project_all(self, data):
        """
        Project every split with the same mean and components.

        data: dict from data.load_splits(). Returns a DROP-IN representation
        dict using identical key names (X_train, y_train, X_val, y_val,
        X_test, y_test), so it can be passed straight to train_classifier in
        place of the raw dict. y_* are carried over unchanged, since reducing the
        features does not change the labels.
        """
        out = {}
        for split in ("train", "val", "test"):
            out[f"X_{split}"] = self.transform(data[f"X_{split}"])
            out[f"y_{split}"] = data[f"y_{split}"]
        out["variance_retained"] = self.variance_retained()
        out["k"] = self.k
        out["representation"] = f"pca_{self.k}"
        return out

    def save(self, path):
        """
        Persist the fitted PCA.

        Atomic, matching run_tracker: a crash mid-write must not leave a
        half-written file that fails to load on resume.
        """
        atomic_save({
            "k": self.k,
            "mean": self.mean,
            "components": self.components,
            "eigenvalues": self.eigenvalues,
            "total_variance": self.total_variance,
        }, path)

    @classmethod
    def load(cls, path, device=None):
        """Restore a fitted PCA. Re-projects identically to the original."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        p = cls(k=ckpt["k"])
        p.mean = ckpt["mean"]
        p.components = ckpt["components"]
        p.eigenvalues = ckpt["eigenvalues"]
        p.total_variance = ckpt["total_variance"]
        return p