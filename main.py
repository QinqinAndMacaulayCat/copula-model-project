"""
test 

"""
import numpy as np
import pandas as pd

from CVine import *
from distribution import Multivariate
from VaR import CCVaR
from data_fetcher import DataFetcher
from GaussianCopula import GaussianCopula



def main():
    # get data
    tickers = ['^GSPC', '^DJI', '^TNX', '^IXIC', '^RUT']
    start_date = '2023-01-01'
    end_date = '2024-11-01'

    dft = DataFetcher(tickers=tickers, 
                      start_date=start_date, 
                      end_date=end_date)
    dft.fetch_and_save_data()
    dft.plot_distribuion()
    y = dft.data
    n_samples = 300

    dataproc = Multivariate(y)
    df_u = dataproc.empircal_cdf()
    u = df_u.values
    # CVine
    #Gaussian
    np.random.seed(6)
    cv = CVine(u, copulaType="Gaussian")
    cv.build_tree()
    cv.fit2()
    simulated_data = cv.simulate(n_samples)
    simulated_data_reversed = dataproc.empircal_ppf(simulated_data)

    factors = simulated_data_reversed[:, :3] 
      
    np.savetxt("result/simulated_data_Gaussian.csv", 
               simulated_data_reversed,
               delimiter=",",
               header=",".join(tickers))

    # CCVaR

    ccvar_model = \
        CCVaR(data=simulated_data_reversed,
              factors=factors, alpha=0.05)
    ccvar_value = \
        ccvar_model.calculate_ccvar(
                target_asset_index=0, 
                factor_index=1)
    print(f"CCVaR of Asset_1 with respect to Factor_2: \
          {ccvar_value:.4f}")

    #Clayton
    cv_clayton = CVine(u, copulaType="Clayton")
    cv_clayton.build_tree()
    cv_clayton.fit2()
    simulated_data_clayton = cv_clayton.simulate(n_samples)
    simulated_data_reversed_clayton =\
       dataproc.empircal_ppf(simulated_data_clayton)
    
    np.savetxt("result/simulated_data_Clayton.csv", 
               simulated_data_reversed_clayton, 
               delimiter=",", header=",".join(tickers)) 
    # CCVaR
    factors_clayton =  \
        simulated_data_reversed_clayton[:, :3]
    ccvar_model = \
        CCVaR(data=simulated_data_reversed_clayton, \
          factors=factors_clayton, alpha=0.05)
    ccvar_value = \
        ccvar_model.calculate_ccvar(
            target_asset_index=0,
            factor_index=1)
    print(f"CCVaR of Asset_1 with respect to Factor_2: \
          {ccvar_value:.4f}")
    
    df_simuldata_Gaussian = pd.DataFrame(simulated_data, columns=tickers)
   
    df_simuldata_Clayton = pd.DataFrame(simulated_data_clayton, columns=tickers)

    corr_init = pd.DataFrame(u).corr()
    corr_Gaussian = pd.DataFrame(df_simuldata_Gaussian).corr()
    corr_Clayton = pd.DataFrame(df_simuldata_Clayton).corr()

    # Gaussian Multivariate Copula
    gc = GaussianCopula(y, tickers)
    gc.estimate_paras()
    gc.estimate_corr()
    simulated_data, simulated_data_reversed = gc.generate_samples(n_samples)
    factors = simulated_data_reversed[:, :3]

    np.savetxt("result/simulated_data_GaussianMultivariate.csv",
               simulated_data_reversed,
               delimiter=",",
               header=",".join(tickers))

    # CCVaR
    ccvar_model = \
        CCVaR(data=simulated_data_reversed,
              factors=factors, alpha=0.05)
    ccvar_value = \
        ccvar_model.calculate_ccvar(
            target_asset_index=0,
            factor_index=1)
    print(f"CCVaR of Asset_1 with respect to Factor_2: \
          {ccvar_value:.4f}")

    df_simuldata_GaussianM = pd.DataFrame(simulated_data, columns=tickers)
    corr_GaussianM = pd.DataFrame(df_simuldata_GaussianM).corr()
    dataproc.heatmap(corr_init, "Correlation_Matrix_of_Initial_Data")
    dataproc.heatmap(corr_Gaussian, "Correlation_Matrix_of_Gaussian_Copula_CVine")
    dataproc.heatmap(corr_Clayton, "Correlation_Matrix_of_Clayton_Copula_CVine")
    dataproc.heatmap(corr_GaussianM, "Correlation_Matrix_of_GaussianM")

    dataproc.plot_kde_comparison(df_u, title="Compare_CDFs_of_Initial_Data")
    dataproc.plot_kde_comparison(df_simuldata_Gaussian, title="Compare_CDFs_of_Gaussian_Copula_CVine")
    dataproc.plot_kde_comparison(df_simuldata_Clayton, title="Compare_CDFs_of_Clayton_Copula_CVine")
    dataproc.plot_kde_comparison(df_simuldata_GaussianM, title="Compare_CDFs_of_Gaussian_Copula")

    ## show extreme value correlation
    corr_extreme_init = dataproc.extreme_value_correlation(y, percentile=5, direction="lower")
    corr_extreme_Gaussian = dataproc.extreme_value_correlation(df_simuldata_Gaussian, percentile=5, direction="lower")
    corr_extreme_Clayton = dataproc.extreme_value_correlation(df_simuldata_Clayton, percentile=5, direction="lower")
    corr_extreme_GaussianM = dataproc.extreme_value_correlation(df_simuldata_GaussianM, percentile=5, direction="lower")


    with open("result/lower_correlation.txt", "w") as file:
        file.write(f"Initial Data\n")  # Write header
        corr_extreme_init.to_string(file)       # Write DataFrame
        file.write("\n\n")  
        file.write(f"Gaussian Copula Cvine\n")
        corr_extreme_Gaussian.to_string(file)
        file.write("\n\n")
        file.write(f"Clayton Copula Cvine\n")
        corr_extreme_Clayton.to_string(file)
        file.write("\n\n")
        file.write(f"Gaussian Copula\n")
        corr_extreme_GaussianM.to_string(file)
        file.write("\n\n")

if __name__ == "__main__":
    main()
