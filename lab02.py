from tools import *


class ProdLayer:
    """ Template example for a class implementing a single layer.
        This example performs coordinate-wise multiplication. """
    
    def forward(self, x: np.array, w: np.array) -> np.array:
        """
        :param x: [N x d] -- layer input of N data points with d dimensions
        :param w: [d] -- d weights
        :return: layer output y = x * w
        """
        self.x = x # remember x
        self.w = w # remember w
        return x * w[np.newaxis, :] # multiply each data point features coordinate-wise with w
    
    def backward(self, dy: np.array) -> Tuple[np.array, np.array]:
        """
        :param dy: [N x d] -- gradient with respect to the layer output
        :return: tuple of gradients in the layer inputs (dx, dw)
        """
        dx = dy * self.w[np.newaxis, :]
        dw = (dy * self.x).sum(axis=0)
        return (dx, dw)


class HiddenLinearLayer():
    """ Template for a class implementing a hidden layer.
        This template performs linear mapping y = W * x + b """

    def forward(self,x:np.array,w:np.array,b:np.array) -> np.array:
        """
        :param x: [N x input_size] -- layer input of N data points with k dimensions
        :param w: [input_size x output_size] -- k weights
        :param b: [output_size] -- offset 
        :return: layer output y = x * W + b  y: [N * output_size]
        """
        #remember
        self.x = x
        self.w = w
        self.b = b

        y =  np.einsum('ij,jk',x,w) + b

        return y

    def backward(self,dy:np.array) -> Tuple[np.array,np.array,float]:
        """
        :param dy: [N x output_size] -- gradient with respect to the layer output
        :return: tuple of gradients in the layer inputs (dx, dw, db)
        """
        dx = dy @ self.w.T

        dw = self.x.T @ dy
        
        db = np.sum(dy,axis=0)

        return tuple([dx,dw,db])


class SumLayer:
    """ Another example, implementing summation of all features operation """

    def forward(self, x: np.array) -> np.array:
        """
        :param x: [N x d] -- layer input of N data points with d dimensions
        :return y: [N] -- sum of x along dimension 1
        """
        self.x = x # remember x
        y = x.sum(axis=1)
        return y

    def backward(self, dy: np.array) -> np.array:
        """
        :param dy: [N] -- gradient with respect to the layer output
        :return: gradients in the layer input (dx) [N x d]
        """
        dx = np.empty_like(self.x)
        dx[:,:] = dy[:,np.newaxis]
        return dx


class ReLuLayer:
    """ Layer implementing ReLu function (which is one of activation functions)"""

    def forward(self,x: np.array)-> np.array:
        """
        :param x: [N x input_size] -- layer input of N data points with d dimensions
        :return y: [N x input_size] -- layer output y = max(x,0)
        """
        self.x = x #remember x

        y = np.maximum(x,0)

        return y

    def backward(self,dy: np.array)-> np.array:
        """
        :param dy: [N x output_size] -- gradient with respect to the layer output
        :return: gradients in the layer input (dx) [N x output_size]
        """
        dx = dy * ( (self.x >= 0) * 1 )

        return dx


class MSELoss:
    """ Template example for the loss function. This example shows Mean Squared Error loss, suitable for regression. """
    
    def forward(self, x: np.array, y: np.array) -> float:
        """
        :param x: [N] -- scores of N data points
        :param y: [N] -- target labels
        :return: loss [] -- scalar loss
        """
        self.delta = x - (2 * y - 1)  # compute and remember difference
        loss = (self.delta ** 2).mean()
        return loss
    
    def backward(self) -> np.array:
        """
        :input: no input to the top gradient function
        :return: gradient in the layer input dx
        (gradient in target y is not needed)
        """
        dx = 2 / self.delta.shape[0] * self.delta
        return dx

class SigmoidAndNLLossLayer:
    """Implements log-likelihood assuming activation layer is sigmoid layer"""

    def forward(self,x: np.array,y:np.array)-> float:
        """
        :param x: [N]  -- output of last linear layer
        :param y: [N]  -- target labels
        :return: loss [] -- scalar loss
        """
        #remember 
        y = np.array(y).reshape(-1,1)
        self.y = y
        self.x = x
        
        delta = np.log(1+ np.exp( np.power(-1,y) * x) )         
        loss = delta.mean()
        
        return loss
    
    def backward(self)-> np.array:
        """
        :input: no input to the top gradient function
        :return: gradient in the layer input dx [N]
        (gradient in target y is not needed)
        """
        alter = np.power(-1,self.y)
        
        dx = alter*np.exp( alter * self.x)/ (self.x.shape[0]) * (1/(1+np.exp( alter * self.x)))

        return dx.reshape(-1,1)


class MyNet:
    """ Template example for the network """
    
    def __init__(self, input_size, hidden_size):
        # name is needed for printing
        self.name = f'Net-example-hidden-{hidden_size}'
        # define arrays that will be parameters of the network
        self.params = dotdict()  # same as dict but with additional access through dot notation
        self.params.w1 = 2*np.random.rand(input_size,hidden_size) - 1 #shift initial guess into [-1,1]
        self.params.b1 = 2*np.random.rand(1,hidden_size) - 1 #shift initial guess int
        self.params.w3 = 2*np.random.rand(hidden_size,1) - 1 
        self.params.b3 = 2*np.random.rand(1,1) - 1
        # define some layers
        self.layer1 = HiddenLinearLayer()
        self.layer2 = ReLuLayer()
        self.layer3 = HiddenLinearLayer()
        self.loss = SigmoidAndNLLossLayer()
    
    def score(self, x: np.array) -> np.array:
        """
        Return log odds (logits) of predictive probability p(y|x) of the network
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] predicted scores of class 1 for all points
        """
        x1 = self.layer1.forward(x, self.params.w1,self.params.b1)
        
        x2 = self.layer2.forward(x1)
        
        s = self.layer3.forward(x2,self.params.w3,self.params.b3)
        # print("x0:",x)
        # print("y1:",x1)
        # print("y2:",x2)
        # print("s:",s)

        return s
    
    def classify(self, x: np.array) -> np.array:
        """
        Make class prediction for the given gata
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] class 0 or 1 per input point
        """
        return self.score(x).reshape(-1) > 0
    
    def mean_loss(self, x, y):  # forward at training time
        """
        Compute the total loss on the training data
        *
        :param train_data: tuple(x,y)
        x [N x d] np.array data points
        y [N] np.array their classes
        :return: total loss to optimize
        """
        s = self.score(x)

        return self.loss.forward(s, y)

    def backward(self):
        """
        Compute gradients in all parameters
        *
        :return: dict mapping parameters to their gradients for grad descent
        """
        grads = dotdict()
        # backprop loss
        grads.score = self.loss.backward()
        
        # backprop layer 3 (Linear N*hidden_size -> N*1)
        grads.x3, grads.w3,grads.b3 = self.layer3.backward(grads.score)
        
        # backprop layer 2 (ReLu)
        
        grads.x2 = self.layer2.backward(grads.x3)
        
        # backprop layer 1 (product)
        _, grads.w1,grads.b1 = self.layer1.backward(grads.x2)

        # print("Grads of score:",grads.score)
        # print("Grads of input in last layer",grads.x3)
        # print("Grads of weights in last layer",grads.w3)
        # print("Grads of offsets in last layer",grads.b3)
        # print("Grads of Relu",grads.x2)
        return grads

    def check_grad(self, train_data, epsilon=0.0001):
        #see homework assignment
        x,y = train_data

        L = self.mean_loss(x, y)
        grads = self.backward()

        for k in self.params.keys():
            #compute size of epsilon
            n = self.params[k].size

            self.params[k] += epsilon
            Lp = self.mean_loss(x, y)
            self.params[k] -= 2*epsilon
            Lm = self.mean_loss(x, y)

            self.params[k] +=epsilon #restore initial value

            delta = (Lp - Lm ) / 2 
            prod = np.sum(grads[k]*epsilon)
            
            print("Grad in {} error {} o({}):".format(k, delta - prod, (n*(epsilon**2)**1/2) ) )

    def train(self, train_data, epochs=100, step_size=0.1):
        """
        Train the model using gradient descent
        *
        :param train_data: tuple (x,y) of trianing data
        :param epochs: number of epochs for gradient descent
        :param step_size: step size in gradient descent
        """
        x, y = train_data
        for epoch in range(epochs):
            # compute loss of the train data
            L = self.mean_loss(x, y)
            # compute gradients in all parameter
            grads = self.backward()
            # make a grad descent step
            for (k, p) in self.params.items():
                p -= step_size * grads[k]
                # print current loss, should be going down
            print(L)



if __name__ == "__main__":

    # use provided class to generate data, visualize data and decision boundaries
    model = G2Model()
    train_data = model.generate_sample(500)
    test_data = model.generate_sample(10000)

    
    # create our network
    net = MyNet(2, hidden_size=500)

    # for k in net.params.keys():
    #     print(k,net.params[k])

    net.check_grad(train_data)
    input("Press enter to continue")
    print("Training net:")
    net.train(train_data, epochs=100, step_size=0.1)

    err_rate = model.test_error(net, test_data)
    print(f"Achieved error rate:{err_rate * 100:.3f}%")
    err_rate0 = model.test_error(model, test_data)
    print(f"GT error rate:{err_rate0 * 100:.3f}%")

    #
    print(f"Drawing -- see {net.name}.pdf")
    model.plot_boundary(train_data, net)


