"""
test 

"""

import numpy as np
import pandas as pd
from CVine import *
from distribution import Multivariate
from VaR import CCVaR


def main():
    np.random.seed(0)
    y1 = np.random.normal(0, 1, 100)
    y2 = np.random.normal(0, 1, 100)
    y3 = np.random.normal(0, 1, 100)
    y4 = np.random.normal(0, 1, 100)
    y5 = np.random.normal(0, 1, 100)

    y = pd.DataFrame({"y1": y1, 
                      "y2": y2,
                      "y3": y3,
                      "y4": y4,
                      "y5": y5})
    dataproc = Multivariate(y)
    u = dataproc.empircal_cdf()
    u = u.values
    cv = CVine(U=u, copulaType="Clayton", max_depth=2)

    cv.build_tree()
    cv.fit2()
    simulated_data = cv.simulate(1000)
    print("simulated", simulated_data[simulated_data >1])
    # print(simulated_data)
    # print(cv.tree["thetaMatrix"])
    # print(pd.DataFrame(simulated_data).corr())

    factors = simulated_data[:, :3]   
    ccvar_model = CCVaR(data=simulated_data, factors=factors, alpha=0.05)
    ccvar_value = ccvar_model.calculate_ccvar(target_asset_index=0, factor_index=1
                                                )
    print(f"CCVaR of Asset_1 with respect to Factor_2: {ccvar_value:.4f}")

        
if __name__ == "__main__":
    main()
