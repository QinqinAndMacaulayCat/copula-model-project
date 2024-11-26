"""
test 

"""

import numpy as np
import pandas as pd
from CVine import *
from distribution import Multivariate
from VaR import CCVaR
from data_fetcher import DataFetcher


def main():
    
    tickers = ['^GSPC', '^DJI', '^TNX', '^IXIC', '^RUT']
    start_date = '2023-01-01'
    end_date = '2024-11-01'

    dft = DataFetcher(tickers=tickers, start_date=start_date, end_date=end_date)
    dft.fetch_and_save_data()
    dft.plot_distribuion()
    y = dft.data
    dataproc = Multivariate(y)
    u = dataproc.empircal_cdf()
    u = u.values
    cv = CVine(u, copulaType="Gaussian")
    cv.build_tree()
    cv.fit()
    simulated_data = cv.simulate(1000)
    # todo: check the overflow
    print(cv.tree["thetaMatrix"])
    # print(pd.DataFrame(simulated_data).corr())

    simulated_data_reversed = dataproc.empircal_ppf(simulated_data)
    factors = simulated_data[:, :3]   
    ccvar_model = CCVaR(data=simulated_data, factors=factors, alpha=0.05)
    ccvar_value = ccvar_model.calculate_ccvar(target_asset_index=0, factor_index=1
                                                )
    print(f"CCVaR of Asset_1 with respect to Factor_2: {ccvar_value:.4f}")

        
if __name__ == "__main__":
    main()
