import numpy as np
from scipy.optimize import minimize
from copula import Clayton

class CVine(object):
    layer = {"root": [], # list of root nodes. ex. in F(u1, u2|v), v is the root node 
             "parentnode": {}, # index of nodes in last level. ex. {1: (1,2)} means the node 1 in this tree level got from the node pair (1,2) in last level
             "node": [], # index of the nodes. from 0 to l. this is not the actual number index in the original data.
             "pair": [],# list of node pairs in the tree, ex. in F(u1, u2|v), (u1, u2) is a node pair. here the node pair is the index of nodes in the root, which is different from the "node". 
             "level": 0, # level of the tree (k). 0-root, 1-1st level, 2-2nd level, ...
             "nodenum": 0, # number of the nodes in this tree (l). equal to n - k 
             "edgenum": 0, # number of the edges in this tree. equal to l as our node number is the actual number minus1.
             "V": None,  # h functions in this level. V[:, j] is the h function of node j.
             "dependence": [] # match the pair of nodes
             }    

    tree = {"thetaMatrix": None, # copula parameter matrix in this level. it is a upper matrix. thetaMatrix[i, j] is the copula parameter in level j between node 1 and node i+1. 
            "structure": {}, # the tree structure in this level. the key is the node index, the value is layer.
            "depth": 0, # the depth of the tree, 0 means only has root. 
            }
    max_depth = None # if the max_depth is given, the only use the first max_depth layers to fit the model. 

    def __init__(self, U, copulaType="Clayton",dependence=None, max_depth=None):
        
        """
        U: np.array, data matrix. follows uniform distribution
        dependence: np.array, dependence matrix constructed by 0 and 1. dependence[i, j] = 1 means i and j are dependent; 0 means independent. 

        """
        self.U = U
        self.T = U.shape[0]
        self.variable_num = U.shape[1] - 1  # to make the structure more clear, all the variables are indexed from 0. Therefore, when the variable_num is n, we actually have n+1 variables x0, x1, ..., xn.
        self.dependence = dependence if dependence is not None else np.ones((self.variable_num+1, self.variable_num+1)) # if dependence is None, we assume all the variables are dependen . if not, the copula function between independent variables will be 1.
        if copulaType == "Clayton":
            self.copula = Clayton()
        else:
            raise ValueError("The copula type is not supported.")

        self.max_depth = max_depth if max_depth is not None else self.variable_num

        
    def build_tree(self):
        """
        build the tree structure. 
        """
        self.build_root()
        while self.tree["depth"] < self.max_depth:
            self.build_kth_tree()

    def build_root(self):
        """
        build the root of the tree. the root is basically 
        """
        
        layer = self.layer.copy()
        layer["level"] = 0
        layer["V"] = self.U.copy() # the F(x|v) in the first layer is the empirical cdf of x. 
        layer["nodenum"] = self.variable_num
        layer["edgenum"] = self.variable_num 
        layer["node"] = list(range(0, layer["nodenum"] + 1))
        layer["dependence"] = self.dependence[0, :]
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
        (layer["pair"], layer["node"], layer["parentnode"], layer["root"], layer["dependence"]) = self.pair_nodes(last_layer)
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
            dependent = np.empty(last_layer["nodenum"] + 1)
            dependent[0] = 1
            for i in range(1, last_layer["nodenum"] + 1):
                if i > 1: # todo  
                    dependent[i] = 0 if self.dependence[pairs[-1][1], pairs[0][1]] == 0 else 1

            return (pairs, 
                   nodes, 
                   parentnodes, 
                   [],
                   dependent)
        else:
            pairs = []
            parentnodes = {}
            last_pairs = last_layer["pair"]
            
            common_node = last_pairs[0][0] # always set the first node as the center node in each layer. and the common element between pairs are the left element in the center pair. the common element between the center node and neighbor nodes will be set into the new root.

            new_root = last_layer["root"] + [common_node]
            pair_left = last_pairs[0][1] # the right element in the center pair will be the left element in pairs in this layer. 
            dependent = np.empty(last_layer["nodenum"] + 1)
            dependent[0] = 1
            for i in range(1, last_layer["nodenum"] + 1):
                pairs.append(tuple((pair_left, last_pairs[i][1])))
                parentnodes[i-1] = (0, i)  # the i-1th node in this layer is from the pair (0, i) in last layer. ex. the first layer in the second layer is from the node pair (0, 1) in the first layer.   
                if i > 1: # todo  
                    dependent[i] = 0 if self.dependence[pairs[-1][1], pairs[0][1]] == 0 else 1

                
        return (pairs, 
                nodes,
                parentnodes, 
                new_root,
                dependent)


    def fit(self):
        """
        fit the vine tree model by maximizing the likelihood of the whole tree.

        """
        paramNum = sum([self.tree["structure"][layer]["edgenum"] for layer in range(0, self.tree["depth"])])

        thetaParams = np.ones(paramNum)
        bounds = [self.copula.bound] * paramNum

        result = minimize(self.get_likelihood, thetaParams, bounds=bounds)
        thetaMatrix = np.zeros((self.tree["depth"], self.tree["structure"][0]["edgenum"]))
        n = 0
        for i in range(0, self.tree["depth"]):
            for j in range(0, self.tree["structure"][i]["edgenum"]):
                thetaMatrix[i, j] = result.x[n]
                n += 1

        self.tree["thetaMatrix"] = thetaMatrix


    def fit2(self):
        
        """
        fit the parameters through maximizing the likelihood in each layer.
        """
        self.tree["thetaMatrix"] = np.zeros((self.tree["depth"], self.tree["structure"][0]["edgenum"]))

        for i in range(1, self.tree["depth"]+1):
            last_layer = self.tree["structure"][i-1]
            layertheta = np.ones(last_layer["edgenum"])
            bounds = [self.copula.bound] * last_layer["edgenum"]
            result = minimize(self.get_layer_likelihood, layertheta, args=(last_layer, ), bounds=bounds)

            self.tree["thetaMatrix"][i-1, :last_layer["edgenum"]] = result.x

            self.tree["structure"][i]["V"] = self.get_layer_h(result.x, last_layer)



    def simulate(self, n):
        """
        simulate the data from the vine tree model
        param n: int, the number of the data to be simulated for each variable.

        """
        if self.tree["thetaMatrix"] is None:
            print("Please fit the model first.")
            return None
        
        else:

            W = np.random.uniform(0, 1, n * (self.variable_num + 1))
            V = np.empty((n, self.variable_num+1, self.variable_num+1))
            W = W.reshape((n, self.variable_num + 1))
            U = np.empty((n, self.variable_num + 1))
            U[:, 0] = W[:, 0]
            V[:, 0, 0] = W[:, 0] 
            c = min(self.tree["depth"], self.variable_num)
            for i in range(1, self.variable_num + 1):
                V[:, 0, i] = W[:, i]
                for k in range(0, min(i, c)):
                    self.copula.theta = self.tree["thetaMatrix"][k, i-k-1]
                    V[:, 0, i] = self.copula.inverse_h(V[:, 0, i], V[:, k, k])
                
                U[:, i] = V[:, 0, i]

                for j in range(0, min(i,c)): 
                    self.copula.theta = self.tree["thetaMatrix"][j, i-j-1]
                    V[:, j + 1, i] = self.copula.h(V[:, j, i], V[:, j, j])
            
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
            total_likelihood += self.get_layer_likelihood(layertheta, last_layer)
            
            self.tree["structure"][k]["V"] = self.get_layer_h(layertheta, last_layer)

        return total_likelihood

    def get_layer_likelihood(self, thetaParams, last_layer):
        """get the likelihood of the layer"""
        likelihood = 0
        
        for i in range(1, last_layer["nodenum"]+1): # totally l copula functions
            
            self.copula.theta = thetaParams[i-1]
            likelihood += (np.nansum(np.log(self.copula.c(last_layer["V"][:, 0], last_layer["V"][:, i]))) if last_layer["dependence"][i] == 1 else 0)
        
        return -likelihood

    def get_layer_h(self, thetaParams, last_layer):
        """get the h function of the layer"""
        V = np.empty((self.T, last_layer["nodenum"])) # the total nodes of this layer is the number of nodes in last layer minus 1, which is equal to the edges in last layer.
        
        for i in range(1, last_layer["nodenum"]+1):
            self.copula.theta = thetaParams[i-1]
            V[:, i-1] = self.copula.h(last_layer["V"][:, 0], last_layer["V"][:, i])
        
        return V

    

