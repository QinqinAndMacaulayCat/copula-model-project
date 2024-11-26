"""
test 

"""

import numpy as np
import pandas as pd
from CVine import *
from distribution import Multivariate



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
    cv = CVine(u)
    cv.build_tree()
    cv.fit2()
    simulated_data = cv.simulate(1000)
    # print(simulated_data) 
    print(cv.tree["thetaMatrix"])
    print(pd.DataFrame(simulated_data).corr())
    

if __name__ == "__main__":
    main()
