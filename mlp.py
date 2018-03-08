import theano
import numpy as np
import theano.tensor as T
from logsitic_sgd import LogisticRegression, load_data
class hiddenLayer(object):
	def __init__(self,rng,inputs,n_in,n_out,W=None,b=None,activation=T.tanh):
		if (!W=None):
			self.W=W
		else:
			self.W=rng.randn(n_in,n_out)/sqrt(6/(n_in+n_out));
		self.inputs=inputs
		self.output=activation(T.dot(inputs,W)+b)
		self.params=[theano.shared(self.W,name='W'),theano.shared(self.b,name='b')]

class MLP(object):
	def __init__(self,rng,inputs,n_in,n_hidden,n_out):
		self.hiddenLayer=hiddenLayer(rng=rng,inputs=inputs,n_in=n_in,n_out=n_hidden)
		self.logisticLayer=LogisticRegression(rng=rng,inputs=self.hiddenLayer.output,
				n_in=n_hidden,n_out=n_out)
		self.params=self.hiddenLayer.params+self.logisticLayer.params
		self.inputs=inputs
		self.output=self.logisticLayer.output
		self.negative_log_likelihood=self.logisticLayer.negative_log_likelihood

def train(train_x,train_y,valid_x,valid_y,test_x,test_y,opts):
	n_train_batches=np.floor(train_x.shape[0]/opts.batch_size)
	index=T.iscalar()
	x=mlp.inputs[0]
	y=mlp.inputs[1]
	cost=T.sum(mlp.negative_log_likelihood(y))+L2_reg*T.sum(T.pow(mlp.hiddenLayer.W,2))+L2_reg*T.sum(T.pow(mlp.logisticLayer.W,2))
	grad=[theano.grad(cost,param) for param in mlp.params]
	train_model=theano.function(inputs=[index],outputs=[mlp.output],
			updates=[(param,param-opts.learningRate*gparam) for param,gparam in zip(mlp.params,grad)]
			,givens={
				x:train_x[index*opts.batch_size+1:(index+1)*opts.batch_size]
				y:train_y[index*opts.batch_size+1:(index+1)*opts.batch_size]
				}
			)
	
	predict_model=theano.function(inputs=[x,y],outputs=[mlp.output,cost])
	
	for i in range(opts.n_epoches):
		for j in range(n_train_batches):
			train_model(j)
		
			
		
def mlp_test():
	
