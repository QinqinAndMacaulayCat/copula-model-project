import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

class CCVaR:
    def __init__(self, data, n_factors, alpha=0.05):
        """
        Initialize the CCVaR class.
        Parameters:
        data: np.array, asset return matrix (T x N), each column represents one asset.
        n_factors: int, PCA factor numbers.
        alpha: float, percentile threshold (e.g., 0.05 for 5%).
        """
        self.data = data  # Asset returns
        self.n_factors = n_factors  # Number of principal components
        self.factors = self.calculate_pca_factors()  # Risk factors
        self.alpha = alpha  # Percentile threshold
        self.T, self.N = data.shape  # Data dimensions: T = time steps, N = number of assets

    def calculate_pca_factors(self):
        """
        Perform PCA on the asset returns to generate risk factors.
        Returns:
        np.array, PCA-transformed factors (T x n_factors).
        """
        pca = PCA(n_components=self.n_factors)
        pca_factors = pca.fit_transform(self.data)
        return pca_factors

    def _transform_to_uniform(self, x):
        """
        Transform data to [0, 1] based on cumulative distribution function (CDF).
        Parameters:
        x: np.array, raw data.
        Returns:
        np.array, transformed data in [0, 1].
        """
        return rankdata(x, method="average") / len(x)

    def _get_extreme_indices(self, factor_col):
        """
        Get the indices of extreme events based on the given percentile.
        Parameters:
        factor_col: np.array, values of a single risk factor.
        Returns:
        np.array, indices of extreme events.
        """
        factor_cdf = self._transform_to_uniform(factor_col) 
        return np.where(factor_cdf <= self.alpha)[0] 

    def calculate_ccvar(self, target_asset_index, factor_index):
        """
        Calculate the CCVaR for a single asset with respect to a single risk factor.
        Parameters:
        target_asset_index: int, index of the target asset.
        factor_index: int, index of the risk factor.
        Returns:
        float, CCVaR value.
        """
        
        target_returns = self.data[:, target_asset_index]
        factor_col = self.factors[:, factor_index]
        extreme_indices = self._get_extreme_indices(factor_col)
        extreme_returns = target_returns[extreme_indices]
        ccvar_value = np.mean(extreme_returns)

        return ccvar_value

    def calculate_all_ccvar(self):
        """
        Calculate CCVaR for all assets with respect to all risk factors and generate a matrix.
        Returns:
        np.array, CCVaR matrix (N x F), rows represent assets, columns represent risk factors.
        """
        results = np.zeros((self.N, self.n_factors))
        for i in range(self.N):  
            for j in range(self.n_factors):
                results[i, j] = self.calculate_ccvar(i, j) 
        
        return results

    def summarize_results(self):
        """
        Summarize results and output a CCVaR matrix with asset and factor names.
        Returns:
        dict, contains the CCVaR matrix and the corresponding asset and factor labels.
        """
        ccvar_matrix = self.calculate_all_ccvar()
        return {
            "CCVaR Matrix": ccvar_matrix,
            "Assets": [f"Asset_{i+1}" for i in range(self.N)],
            "Factors": [f"Factor_{j+1}" for j in range(self.n_factors)],
        }

    def plot_ccvar(self, asset_names, save=False, path=None):
        """
        Plot and save the CCVaR matrix.
        """
        ccvar_matrix = self.calculate_all_ccvar()
        factor_names = [f"PCAFactor_{i}" for i in range(1, self.n_factors+1)]
        ccvar_df = pd.DataFrame(ccvar_matrix, index=asset_names, columns=factor_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(ccvar_df,
                    annot=True,
                    fmt=".4f",
                    cmap="YlGnBu",
                    linewidths=0.5,
                    linecolor="gray")

        plt.title("CCVaR Heatmap")
        plt.xlabel("Risk Factors")
        plt.ylabel("Assets")

        if save:
            plt.savefig(path+".png", dpi=300, bbox_inches="tight")

            np.savetxt(path+".csv",
                       ccvar_df.T,
                       delimiter=",",
                       header=",".join(asset_names))

        plt.show()

    
if __name__ == "__main__":
    simulated_data = np.random.normal(0, 1, (1000, 5))
    ccvar_model = CCVaR(data=simulated_data, n_factors=3, alpha=0.05)
    ccvar_value = ccvar_model.calculate_ccvar(target_asset_index=0, factor_index=1)
    print(f"CCVaR of Asset_1 with respect to Factor_2: {ccvar_value:.4f}")

    ccvar_matrix = ccvar_model.calculate_all_ccvar()
    print("\nCCVaR Matrix:")
    print(ccvar_matrix)

    asset_names = [f"Asset_{i+1}" for i in range(5)]
    ccvar_model.plot_ccvar(asset_names=asset_names, save=True, path="result/CCVaR_test")






  
