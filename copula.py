import numpy as np
import pandas as pd
from scipy.optimize import minimize 
from distribution import Multivariate


class Clayton:
    theta = 0

    def c(self, u, v):
        """
        return the density of Clayton copula
        """
        return (1 + self.theta) * np.power(u * v, -1 - self.theta) \
            * np.power(np.power(u, -self.theta) + np.power(v, -self.theta) - 1, -2 - 1 / self.theta)

    def h(self, u, v):
        """
        return the h function/partial derivative F(u|v)  of Clayton copula
    
        """
        return np.power(v, -self.theta-1) * np.power(np.power(u, -self.theta) + np.power(v, -self.theta) - 1, -1 - 1/self.theta)
    
    def inverse_h(self, w, v):

        """
        return the inverse of h function, which is the conditional CDF of u given v. 
        """
        
        a = w * np.power(v, self.theta+1) 
        b = np.power(a, -self.theta/(1+self.theta)) + 1 - np.power(v, -self.theta)
        return np.power(b, -1/self.theta)
    
class CVine(Clayton):
    layer = {"root": [], # list of root nodes. ex. in F(u1, u2|v), v is the root node 
             "parentnode": {}, # index of nodes in last level. ex. {1: (1,2)} means the node 1 in this tree level got from the node pair (1,2) in last level
             "node": [], # index of the nodes. from 0 to l. this is not the actual number index in the original data.
             "pair": [],# list of node pairs in the tree, ex. in F(u1, u2|v), (u1, u2) is a node pair. here the node pair is the index of nodes in the root, which is different from the "node". 
             "level": 0, # level of the tree (k). 0-root, 1-1st level, 2-2nd level, ...
             "nodenum": 0, # number of the nodes in this tree (l). equal to n - k 
             "edgenum": 0, # number of the edges in this tree. equal to l as our node number is the actual number minus1.
             "V": None,  # h functions in this level. V[:, j] is the h function of node j.
             }    

    tree = {"thetaMatrix": None, # copula parameter matrix in this level. it is a upper matrix. thetaMatrix[i, j] is the copula parameter in level j between node 1 and node i+1. 
            "structure": {}, # the tree structure in this level. the key is the node index, the value is layer.
            "depth": 0, # the depth of the tree, 0 means only has root. 
            }
               
    def __init__(self, U, dependence=None):
        
        """
        U: np.array, data matrix. follows uniform distribution
        dependence: np.array, dependence matrix constructed by 0 and 1. dependence[i, j] = 1 means i and j are dependent; 0 means independent. 
        """
        self.U = U
        self.T = U.shape[0]
        self.variable_num = U.shape[1] - 1  # to make the structure more clear, all the variables are indexed from 0. Therefore, when the variable_num is n, we actually have n+1 variables x0, x1, ..., xn.
        self.dependence = dependence # if dependence is None, we assume all the variables are dependent. if not, the copula function between independent variables will be 1.
        
    def build_tree(self):
        """
        build the tree structure. 
        """
        self.build_root()
        while self.tree["depth"] < self.variable_num:
            self.build_kth_tree()

    def build_root(self):
        """
        build the root of the tree. the root is basically 
        """
        
        layer = self.layer.copy()
        layer["level"] = 0
        layer["V"] = self.U.copy()
        layer["nodenum"] = self.variable_num
        layer["edgenum"] = self.variable_num 
        layer["node"] = list(range(0, layer["nodenum"] + 1))
        self.tree["structure"][0] = layer

        
    def build_kth_tree(self):
        """
        build the kth tree. 
        """

        if self.tree["depth"] >= self.variable_num:
            print("The tree depth is already the maximum.")

        last_layer = self.tree["structure"][self.tree["depth"]]

        layer = self.layer.copy()
        layer["level"] = last_layer["level"] + 1
        layer["nodenum"] = last_layer["nodenum"] - 1
        layer["edgenum"] = layer["nodenum"]
        layer["node"] = list(range(0, layer["nodenum"]+ 1))
        (layer["pair"], layer["node"], layer["parentnode"], layer["root"]) = self.pair_nodes(last_layer)

        self.tree["structure"][layer["level"]] = layer
        self.tree["depth"] = self.tree["depth"] + 1
    
    def pair_nodes(self, last_layer):
        """
        pair the nodes in this layer. 
        here we use the first node in each level as the new central node and combine it with the root in last level to get the new root. This process is same as the process we show in the report.
        """


        nodes = range(0, last_layer["nodenum"] + 1)

        if last_layer["level"] == 0: # the second layer is not conditional copula, so we just combine the center node with neighor nodes without any condition.
            pair_left = last_layer["node"][0]
            pairs = tuple(zip(last_layer["edgenum"] * [pair_left], last_layer["node"][1:]))
            parentnodes = dict(zip(nodes, pairs))
            return (pairs, 
                    nodes, 
                    parentnodes, 
                    [])
        else:
            pairs = []
            parentnodes = {}
            last_nodes = last_layer["node"]
            last_pairs = last_layer["pair"]
            
            common_node = last_pairs[0][0] # always set the first node as the center node in each layer. and the common element between pairs are the left element in the center pair. the common element between the center node and neighbor nodes will be set into the new root.

            new_root = last_layer["root"] + [common_node]
            pair_left = last_pairs[0][1] # the right element in the center pair will be the left element in pairs in this layer. 

            for i in range(1, last_layer["nodenum"] + 1):
                pairs.append(tuple((pair_left, last_pairs[i][1])))
                parentnodes[i-1] = (0, i)  # the i-1th node in this layer is from the pair (0, i) in last layer. ex. the first layer in the second layer is from the node pair (0, 1) in the first layer.
                
        return (pairs, 
                nodes,
                parentnodes, 
                new_root)


    def fit(self):
        """
        fit the vine tree model

        """
        paramNum = sum([self.tree["structure"][layer]["edgenum"] for layer in range(0, self.tree["depth"])])

        thetaParams = np.ones(paramNum)
        result = minimize(self.get_likelihood, thetaParams, bounds=[(1e-6, np.inf)]*paramNum)
        thetaMatrix = np.zeros((self.tree["depth"], self.tree["structure"][0]["edgenum"]))
        n = 0
        for i in range(0, self.tree["depth"]):
            for j in range(0, self.tree["structure"][i]["edgenum"]):
                thetaMatrix[i, j] = result.x[n]
                n += 1

        self.tree["thetaMatrix"] = thetaMatrix
        
    def simulate(self, n):
        """
        simulate the data from the vine tree model
        """
        if self.tree["thetaMatrix"] is None:
            print("Please fit the model first.")
            return None
        
        else:
            W = np.random.uniform(0, 1, n * (self.variable_num + 1))
            V = np.empty((self.tree["depth"]+1, n, self.variable_num+1))
            W = W.reshape((n, self.variable_num + 1))
            U = np.empty((n, self.variable_num + 1))
            U[:, 0] = W[:, 0]
            V[0, :, 0] = W[:, 0] 
            for i in range(1, self.variable_num + 1):
                V[i, :, 0] = W[:, i]
                for k in range(0, i):
                    self.theta = self.tree["thetaMatrix"][k, i-k-1]
                    V[i, :, 0] = self.inverse_h(V[i, :, 0], V[k, :, k])
                
                U[:, i] = V[i, :, 0]

                for j in range(0, i): 
                    self.theta = self.tree["thetaMatrix"][j, i-j-1]
                    V[i, :, j + 1] = self.h(V[i, :, j], V[j, :, j])
            return U
        

    def get_likelihood(self, thetaParams):
        """get the likelihood of the vine tree model"""
        
        total_likelihood = 0
        left = 0 
        right = 0
        for k in range(1, self.tree["depth"] + 1): # ignore the root layer
            # each layer' c function is determined by the last layer's and this layer's theta. number of theta in each layer is equal to the number of nodes in this layer.
            last_layer = self.tree["structure"][k-1]

            left = right
            right = right + last_layer["edgenum"] 
            layertheta = thetaParams[left:right]
            total_likelihood += self.get_layer_likelihood(last_layer, layertheta)
            
            self.tree["structure"][k]["V"] = self.get_layer_h(last_layer, layertheta)

        return total_likelihood

    def get_layer_likelihood(self, last_layer, thetaParams):
        """get the likelihood of the layer"""
        likelihood = 0
        
        for i in range(1, last_layer["nodenum"]+1): # totally l copula functions

            self.theta = thetaParams[i-1]
            likelihood += np.nansum(np.log(self.c(last_layer["V"][:, 0], last_layer["V"][:, i])))
        
        return -likelihood

    def get_layer_h(self, last_layer, thetaParams):
        """get the h function of the layer"""
        V = np.empty((self.T, last_layer["nodenum"])) # the total nodes of this layer is the number of nodes in last layer minus 1, which is equal to the edges in last layer.
        
        for i in range(1, last_layer["nodenum"]+1):
            self.theta = thetaParams[i-1]
            V[:, i-1] = self.h(last_layer["V"][:, 0], last_layer["V"][:, i])
        
        return V

    

if __name__ == "__main__":
    y1 = np.random.normal(0, 1, 100)
    y = pd.DataFrame({"y1": y1, 
                      "y2": y1,
                      "y3": y1,
                      "y4": y1,
                      "y5": y1})
    dataproc = Multivariate(y)
    u = dataproc.empircal_cdf()
    u = u.values
    cv = CVine(u)
    cv.build_tree()
    cv.fit()
    simulated_data = cv.simulate(1000)
    print(cv.tree["thetaMatrix"])
    
    print(pd.DataFrame(simulated_data).corr(), pd.DataFrame(u).corr())

