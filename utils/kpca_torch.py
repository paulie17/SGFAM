import torch
from torch import nn
from torch.nn import Parameter
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

Tensor = torch.Tensor

def kernel_factory(name: str, param: dict):
    assert name in ["rbf", "linear", "poly", "cosine", "sigmoid"] # Updated list
    kernel = None
    if name == "rbf":
        kernel = GaussianKernelTorch(**param)
    elif name == "poly":
        kernel = PolyKernelTorch(**param)
    elif name == "linear":
        kernel = LinearKernel()
    elif name == "cosine":
        kernel = CosineKernel() # Cosine kernel takes no params (or handles defaults internally)
    elif name == "sigmoid":
        kernel = SigmoidKernel(**param) # Sigmoid takes gamma and r
    return kernel

class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        if Y is None:
            Y = X
        return torch.mm(X.t(), Y)

    def phi_inv(self, x):
        return x

    def phi(self, x):
        return x

class GaussianKernelTorch(nn.Module):
    def __init__(self, sigma2=50.0):
        super(GaussianKernelTorch, self).__init__()
        if type(sigma2) == float:
            self.sigma2 = Parameter(torch.tensor(float(sigma2)), requires_grad=False)
            self.register_parameter("sigma2", self.sigma2)
        else:
            self.sigma2 = sigma2

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        if Y is None:
            Y = X

        def my_cdist(x1, x2):
            x1 = torch.t(x1)
            x2 = torch.t(x2)
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30)
            return res.sqrt_()

        D = my_cdist(X, Y)
        return torch.exp(- torch.pow(D, 2) / self.sigma2)

    def phi_inv(self, x):
        raise NotImplementedError()
    
class CosineKernel(nn.Module):
    def __init__(self):
        super(CosineKernel, self).__init__()

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix using the Cosine kernel.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
            
        # N x M matrix of dot products (X.t() @ Y)
        dot_product = torch.mm(X.t(), Y)
        
        # Norms: ||X||, ||Y||
        X_norm = torch.linalg.norm(X, dim=0, keepdim=True) # 1 x N
        Y_norm = torch.linalg.norm(Y, dim=0, keepdim=True) # 1 x M
        
        # N x M matrix of outer product of norms (||X||.t() @ ||Y||)
        norm_product = torch.mm(X_norm.t(), Y_norm)
        
        # K(x, y) = (x.t @ y) / (||x|| * ||y||)
        # Add a small epsilon for numerical stability
        epsilon = 1e-12 
        K = dot_product / (norm_product.clamp_min(epsilon))
        
        # Clamp to ensure values are within [-1, 1] range (due to floating point errors)
        return K.clamp(-1.0, 1.0) 

    def phi_inv(self, x):
        raise NotImplementedError()

    def phi(self, x):
        raise NotImplementedError()
    
class SigmoidKernel(nn.Module):
    def __init__(self, gamma: float = None, r: float = 1.0):
        super(SigmoidKernel, self).__init__()
        # Scikit-learn default for gamma is 1 / n_features, but here we require it to be passed
        self.gamma = Parameter(torch.tensor(float(gamma)) if gamma is not None else torch.tensor(1.0), requires_grad=False)
        self.r = Parameter(torch.tensor(float(r)), requires_grad=False)
        
    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix using the Sigmoid (Tanh) kernel.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
            
        # Inner term: gamma * (X.t() @ Y) + r
        inner_term = self.gamma * torch.mm(X.t(), Y) + self.r
        
        # K(x, y) = tanh(inner_term)
        return torch.tanh(inner_term)

    def phi_inv(self, x):
        raise NotImplementedError()

    def phi(self, x):
        raise NotImplementedError()

class PolyKernelTorch(nn.Module):
    def __init__(self, d: int, t=1.0) -> None:
        super().__init__()
        self.d = d
        self.c = t

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        if Y is None:
            Y = X
        return torch.pow(torch.matmul(X.t(), Y) + self.c, self.d)

    def phi(self, x):
        raise NotImplementedError()

    def phi_inv(self, x):
        raise NotImplementedError()

class KernelPCA(nn.Module):
    """Kernel Principal Component Analysis (KPCA) implemented in PyTorch."""

    def __init__(self, n_components: int, kernel_name: str, kernel_params: dict):
        super().__init__()
        self.n_components = n_components
        self.kernel_func = kernel_factory(kernel_name, kernel_params)
        
        self.alphas: Optional[Tensor] = None
        self.lambdas: Optional[Tensor] = None
        self.X_fit: Optional[Tensor] = None
        self.is_fitted = False

    def center_kernel_matrix(self, K: Tensor) -> Tensor:
        N = K.shape[0]
        J = torch.full((N, N), 1/N, dtype=K.dtype, device=K.device)
        K_c = K - torch.matmul(J, K) - torch.matmul(K, J) + torch.matmul(torch.matmul(J, K), J)
        return K_c

    def fit(self, X: Tensor) -> 'KernelPCA':
        X_t = X.t()
        N = X_t.shape[1]

        K = self.kernel_func(X_t)
        K_c = self.center_kernel_matrix(K)

        # Add small regularization for numerical stability
        reg_term = 1e-6 * torch.eye(K_c.shape[0], dtype=K_c.dtype, device=K_c.device)
        K_c = K_c + reg_term

        # Try CUDA computation, fall back to CPU if it fails
        try:
            lambdas_all, alphas_all = torch.linalg.eigh(K_c)
        except RuntimeError as e:
            if 'cusolver' in str(e).lower():
                print("CUDA eigendecomposition failed, falling back to CPU...")
                K_c_cpu = K_c.cpu()
                lambdas_all, alphas_all = torch.linalg.eigh(K_c_cpu)
                lambdas_all = lambdas_all.to(X.device)
                alphas_all = alphas_all.to(X.device)
            else:
                raise e

        idx = torch.argsort(lambdas_all, descending=True)
        self.lambdas = lambdas_all[idx][:self.n_components]
        self.alphas = alphas_all[:, idx][:, :self.n_components]

        mask = self.lambdas.abs() > 1e-12 
        norm_factor = torch.zeros_like(self.lambdas)
        norm_factor[mask] = 1.0 / torch.sqrt(self.lambdas[mask].abs())
        
        self.alphas = self.alphas * norm_factor

        self.X_fit = X_t
        self.is_fitted = True
        return self

    def transform(self, X: Tensor) -> Tensor:
        if not self.is_fitted:
            raise RuntimeError("KPCA must be fitted before calling transform.")

        X_t = X.t()
        N_new = X_t.shape[1]
        
        K_test = self.kernel_func(X_t, self.X_fit)

        N_train = self.X_fit.shape[1]
        J_new_train = torch.full((N_new, N_train), 1/N_train, dtype=K_test.dtype, device=K_test.device)
        J_train_train = torch.full((N_train, N_train), 1/N_train, dtype=K_test.dtype, device=K_test.device)
        
        K_train = self.kernel_func(self.X_fit)
        
        K_c_test = K_test - torch.matmul(J_new_train, K_train) - \
                   torch.matmul(K_test, J_train_train) + \
                   torch.matmul(torch.matmul(J_new_train, K_train), J_train_train)

        X_transformed = torch.matmul(K_c_test, self.alphas)
        
        return X_transformed

# --------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Data Generation ---
    # Generate concentric circles data: non-linearly separable in 2D
    X_np, y_np = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=50)
    X = torch.tensor(X_np, dtype=torch.float32)
    
    # We will project to 2 components to see if separation is achieved in a new 2D space

    # --- 1. Linear PCA (torch.pca_lowrank) ---
    # Centering the data first for accurate PCA comparison
    X_centered = X - X.mean(dim=0)
    
    # Linear PCA finds the singular vectors V (principal directions)
    _, _, V = torch.pca_lowrank(X_centered, q=2)
    # Projection: X_proj @ V
    X_linear_pca = torch.matmul(X_centered, V) 
    X_linear_pca_np = X_linear_pca.numpy()
    print("Linear PCA: Completed")

    # --- 2. Kernel PCA (RBF Kernel) ---
    kpca = KernelPCA(
        n_components=2, 
        kernel_name='rbf', 
        kernel_params={'sigma2': 0.1} # Adjusted for better 2D separation
    )
    kpca.fit(X)
    X_kpca = kpca.transform(X)
    X_kpca_np = X_kpca.numpy()
    print("Kernel PCA: Completed")

    # --- 3. Visualization and Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Original Data Plot ---
    axes[0].scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], label='Class 0', color='blue', alpha=0.6)
    axes[0].scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], label='Class 1', color='red', alpha=0.6)
    axes[0].set_title('Original Data (2D)')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].legend()
    axes[0].set_aspect('equal', adjustable='box')
    
    # --- Linear PCA Projection Plot ---
    # Plotting the 2D projection from Linear PCA
    axes[1].scatter(X_linear_pca_np[y_np == 0, 0], X_linear_pca_np[y_np == 0, 1], 
                    label='Class 0', color='blue', alpha=0.6)
    axes[1].scatter(X_linear_pca_np[y_np == 1, 0], X_linear_pca_np[y_np == 1, 1], 
                    label='Class 1', color='red', alpha=0.6)
    axes[1].set_title('Linear PCA Projection (2D)')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend()
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].text(0.5, 0.9, 'Classes are highly mixed', 
                 horizontalalignment='center', transform=axes[1].transAxes, color='black', fontsize=10)
    
    # --- Kernel PCA Projection Plot ---
    # Plotting the 2D projection from Kernel PCA
    axes[2].scatter(X_kpca_np[y_np == 0, 0], X_kpca_np[y_np == 0, 1], 
                    label='Class 0', color='blue', alpha=0.6)
    axes[2].scatter(X_kpca_np[y_np == 1, 0], X_kpca_np[y_np == 1, 1], 
                    label='Class 1', color='red', alpha=0.6)
    axes[2].set_title('Kernel PCA Projection (2D)')
    axes[2].set_xlabel('Kernel Principal Component 1')
    axes[2].set_ylabel('Kernel Principal Component 2')
    axes[2].legend()
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].text(0.5, 0.9, 'Classes are clearly separated', 
                 horizontalalignment='center', transform=axes[2].transAxes, color='black', fontsize=10)


    plt.tight_layout()
    plt.savefig('kpca_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to kpca_comparison.png")