# cd pythonprojects
# python3 stats_regress.py



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from statsmodels.graphics.tsaplots import plot_acf
from time import time

data = np.array([[0.1576, 0.4387, 0.8407, 0.1622, 5.39  ],
       [0.9706, 0.3816, 0.2543, 0.7943, 3.9071],
       [0.9572, 0.7655, 0.8143, 0.3112, 3.6193],
       [0.4854, 0.7952, 0.2435, 0.5285, 4.1959],
       [0.8003, 0.1869, 0.9293, 0.1656, 1.6359],
       [0.1419, 0.4898, 0.35  , 0.602 , 6.4107],
       [0.4218, 0.4456, 0.1966, 0.263 , 4.0398],
       [0.9157, 0.6463, 0.2511, 0.6541, 3.8689],
       [0.7922, 0.7094, 0.616 , 0.6892, 4.8914],
       [0.9595, 0.7547, 0.4733, 0.7482, 5.7082],
       [0.6557, 0.276 , 0.3517, 0.4505, 4.5901],
       [0.0357, 0.6797, 0.8308, 0.0838, 4.452 ],
       [0.8491, 0.6551, 0.5853, 0.229 , 3.3803],
       [0.934 , 0.1626, 0.5497, 0.9133, 4.9433],
       [0.6787, 0.119 , 0.9172, 0.1524, 2.0151],
       [0.7577, 0.4984, 0.2858, 0.8258, 6.0634],
       [0.7431, 0.9597, 0.7572, 0.5383, 6.1009],
       [0.3922, 0.3404, 0.7537, 0.9961, 5.4558],
       [0.6555, 0.5853, 0.3804, 0.0782, 5.6223],
       [0.1712, 0.2238, 0.5678, 0.4427, 4.7387],
       [0.706 , 0.7513, 0.0759, 0.1067, 2.5669],
       [0.0318, 0.2551, 0.054 , 0.9619, 5.5655],
       [0.2769, 0.506 , 0.5308, 0.0046, 2.3505],
       [0.0462, 0.6991, 0.7792, 0.7749, 4.8978],
       [0.0971, 0.8909, 0.934 , 0.8173, 6.7528],
       [0.8235, 0.9593, 0.1299, 0.8687, 7.195 ],
       [0.6948, 0.5472, 0.5688, 0.0844, 3.6018],
       [0.3171, 0.1386, 0.4694, 0.3998, 5.7609],
       [0.9502, 0.1493, 0.0119, 0.2599, 4.7579],
       [0.0344, 0.2575, 0.3371, 0.8001, 2.832 ]])


# everything is based on selected columns, then instantiated. 

# find a way to parallelize everything. 
# check out this: https://stackoverflow.com/questions/5909873/how-can-i-pretty-print-ascii-tables-with-python
# it is for nice print boxes.
# find distributions/ tabulated values in python for pillai's trace, Hotelling-Lowling root, and Roy's largest root etc

# beta must be run first for the sake of 

class regression_model:
	# this is all good for multivariate regression.
	def __init__(self,predictor_matrix,response):
		if not isinstance(predictor_matrix,np.ndarray) or not isinstance(response,np.ndarray):
			raise Exception('Data and response must be numpy arrays.')
		predictor_shape = predictor_matrix.shape
		if (len(response.shape) == 1) or (response.shape[0] != predictor_shape[0]):
			raise Exception('Response vector needs to be 2-dimensional and same length as predictor matrix.')
		self.num_obs = predictor_shape[0]
		self.num_predictors = predictor_shape[1]
		self.num_responses = response.shape[1]
		self.X = np.c_[np.ones((predictor_shape[0],1)),predictor_matrix]
		self.XTXinverse = np.linalg.inv(self.X.T.dot(self.X)) # check if there's a better way.
		self.rank = self.num_predictors + 1
		self.response = response

	def compute_beta(self):
		# (num of predictors + 1) X (# of response variables)
		self.Beta_hat = self.XTXinverse.dot(self.X.T).dot(self.response)
		return self.Beta_hat
		# add functionality for stochastic gradient descent minibatch etc

	def compute_predictions(self):
		# (num of obs) X (num of responses)
		self.predicted_y = self.X.dot(self.compute_beta())
		return self.predicted_y

	def compute_errors(self):
		# (num of obs) X (num of responses)
		self.errors = self.response - self.compute_predictions()
		return self.errors
		

	def compute_variance(self,type_='unbiased'):
		# (# of response variables) X (# of response variables)
		errors = self.compute_errors()
		if type_ == 'mle':
			self.var = errors.T.dot(errors)/self.num_obs
		elif type_ == 'unbiased':
			self.var = errors.T.dot(errors)/(self.num_obs - self.rank)
		else:
			raise Exception('Only \'mle\' or \'unbiased\' are valid value for \'type_\'')
		return self.var[0]

	# class method to select new columns	

	 
			


class model_checking(regression_model):

	def __init__(self,predictor_matrix,response):
		super().__init__(predictor_matrix,response)

	def check_residual_normality(self,graph=True):
		# add confidence bands for qq line
		projection_matrix = self.X.dot(np.linalg.inv(self.X.T.dot(self.X))).dot(self.X.T)
		studentized_errors = (super().compute_errors()-0)/pow(super().compute_variance()*(1-np.diag(projection_matrix).reshape(-1,1)),0.5)
		studentized_errors.sort()
		theoretical_errors = stats.norm.ppf((np.arange(1,self.num_obs+1)-0.5)/self.num_obs)
		lins_concordance_correlation_coef = (2*np.cov(np.c_[theoretical_errors,studentized_errors],rowvar=False,ddof=0)[0,1])/(np.var(theoretical_errors)+np.var(studentized_errors))
		R_squared = stats.pearsonr(theoretical_errors.flatten(),studentized_errors.flatten())[0]**2
		if graph:
			fig,ax = plt.subplots()
			ax.scatter(theoretical_errors,studentized_errors,marker='o',facecolors='none',edgecolors='r') 
			line = mlines.Line2D([0, 1], [0, 1], color='blue')
			transform = ax.transAxes
			line.set_transform(transform)
			ax.add_line(line)
			ax.set_aspect('equal')
			ax.set_xlabel('Theoretical Quantiles')
			ax.set_ylabel('Observed Quantiles')
			ax.set_title('QQ Plot for Studentized Residuals')
			
			ax.text(0.05, 0.95, 
				'\n'.join((
					r'$R^2 = {}$'.format(round(R_squared,3)),
					r"$\rho_{\mathrm{crnd}} = %.3f$"%(round(lins_concordance_correlation_coef,3)) 
						)), 
				transform=ax.transAxes, 
				fontsize=14,
        		va='top', 
        		bbox=dict(
        			boxstyle='square',
        			facecolor='white'
        		))
			plt.show()
			return None
		else:
			return lins_concordance_correlation_coef,R_squared

		# add functionality for multivariate errors

	def residual_predictor_relationship(self,transformed_predictor_s,title='Testing For a Need of Extra Terms'):
		# number of responses plots each error var in vector and predictor transformation until a point where there's too much
		# Predictor vs residual: if there is a interaction relationship, add it to the model. Regular predictors vs residuals: check if variance is constant and looks independent.
		fig,ax = plt.subplots()
		ax.scatter(transformed_predictor_s,super().compute_errors(),marker='o',facecolors='none',edgecolors='g')
		ax.set_title(title)
		ax.set_xlabel('Transformed Predictor')
		ax.set_ylabel('Observed Residuals')
		plt.show()

	def residual_prediction_relationship(self,title='Testing for Linearity & Homoscedasticity'):
		# #number of responses plots (1-1 between each response error in vector ) until there's too much
		# Residuals vs predicted value check if variance is constant, and whether relationship looks independent.
		fig,ax = plt.subplots()
		ax.scatter(super().compute_predictions(),super().compute_errors(),marker='o',facecolors='none',edgecolors='g')
		ax.set_title(title)
		ax.set_xlabel('Predictions')
		ax.set_ylabel('Observed Residuals')
		plt.show()

	def residual_time_effects(self,alpha):
		fig,ax = plt.subplots()
		plot_acf(super().compute_errors(),ax=ax,alpha=alpha,unbiased=True,fft=True,title=False,vlines_kwargs=dict(colors='r'))
		ax.set_title('Autocorrelation of Residuals')
		ax.set_xlabel('Lag')
		ax.set_ylabel('Correlation')
		plt.show()

	def scatter_matrix(self, plot_ = False, selection=None):
		pass



class model_analysis(regression_model):

	def __init__(self,predictor_matrix,response,alpha):
		super().__init__(predictor_matrix,response)
		self.alpha = alpha
		# add extra instance variable for verbosity?

	def SSE(self):
		# check if self.errors exist otherwise compute
		errors = super().compute_errors()
		return errors.T.dot(errors)[0][0]

	def confidence_bands_sigma(self):
		# assumes the errors are iid normal with the same variance and zero mean.
		tail1, tail2 = 1-self.alpha/2, self.alpha/2
		W1, W2 = stats.invgamma.ppf(tail1,a=(self.num_obs-self.rank)/2,loc=0,scale=0.5), stats.invgamma.ppf(tail2,a=(self.num_obs-self.rank)/2,loc=0,scale=0.5)
		squared_errors = self.SSE()
		self.confidence_band_sigma = np.array([squared_errors*W2,squared_errors*W1])
		return self.confidence_band_sigma
		# return confidence band for covariance matrix.

	def confidence_interval_t_test_beta(self,num_comparisons=None,test_if_not_zero=False):
		# assumes the errors are iid normal with the same variance and zero mean.
		# bounds can get inflated if there's high multicollinearity (high correlation between features)
		# 
		beta_hat = super().compute_beta()
		if beta_hat.shape[1]>1:
			raise Exception('This is a one-at-a-time (Bonferroni) interval for the components of \'self.compute_beta\' with response one.')
		if (num_comparisons > self.num_predictors+1) or (num_comparisons<1):
			raise Exception('Maximum number of multiple comparisons is number of predictors + 1 (bias term). Minumum number of multiple comparisons is one.')
		
		if num_comparisons == None:
			num_comparisons = self.num_predictors+1
		std_beta_hat = pow(super().compute_variance('unbiased')*np.diag(self.XTXinverse).reshape(-1,1),0.5)
		if not test_if_not_zero:
			W = stats.t.ppf(1-self.alpha/(2*num_comparisons),df=self.num_obs-self.rank)*std_beta_hat
			return np.c_[beta_hat-W,beta_hat+W]
		else:
			# returns p-values testing whether or not each coefficient is zero.
			return stats.t.sf(abs(beta_hat/std_beta_hat),df=self.num_obs-self.rank)

	def confidence_interval_expectation(self,z0):
		# works only for one novel observation. 
		# assumes that the errors are iid normal with same variance and zero mean.
		if not isinstance(z0,np.ndarray):
			raise Exception('New observation vector must be a numpy array.')

		z0 = np.append(1,z0).reshape(1,-1)
		expectation_estimate = z0.dot(super().compute_beta())[0][0]
		W = stats.t.ppf(1-self.alpha/2,df=self.num_obs-self.rank)*pow(z0.dot(self.XTXinverse).dot(z0.T)*super().compute_variance('unbiased'),0.5)[0][0]
		return [expectation_estimate-W,expectation_estimate+W]

	def prediction_interval(self,z0):
		# works only for one novel observation. 
		# assumes that the errors are iid normal with same variance and zero mean.
		if not isinstance(z0,np.ndarray):
			raise Exception('New observation vector must be a numpy array.')

		z0 = np.append(1,z0).reshape(1,-1)
		expectation_estimate = z0.dot(super().compute_beta())[0][0]
		W = stats.t.ppf(1-self.alpha/2,df=self.num_obs-self.rank)*pow((z0.dot(self.XTXinverse).dot(z0.T)+1)*super().compute_variance('unbiased'),0.5)[0][0]
		return [expectation_estimate-W,expectation_estimate+W]

	def test_for_regression_relation(self): # add functionality for subsets. Would need to deal with storing/ somehow not computing again XTX inverse
		# null hypothesis is that all coefficients except bias term are zero
		# alternative is at least one of them not zero
		SSE = self.SSE()
		df1 = self.num_predictors
		df2 = self.num_obs - self.rank
		MSE = SSE/df2
		numobs = self.num_obs
		SSTO = self.response.T.dot(np.eye(numobs)-np.ones((numobs,numobs))/numobs).dot(self.response)
		MSR = (SSTO-SSE)/(df1)
		F = MSR/MSE
		return stats.f.sf(F,df1,df2)[0][0]

	def R_squared(self,adjusted=False):
		# how close responses are to predicted values.
		if not adjusted:
			SSE = self.SSE()
			numobs = self.num_obs
			SSTO = self.response.T.dot(np.eye(numobs)-np.ones((numobs,numobs))/numobs).dot(self.response)
			return float(1 - (SSE/SSTO))
		else:
			# adding more predictors can only increase R^2 and never reduce it. The adjusted R^2 
			# penalizes/offsets for models with a large number of predictors that don't improve much (improving: lowering SSE)
			SSE = self.SSE()
			numobs = self.num_obs
			SSTO = self.response.T.dot(np.eye(numobs)-np.ones((numobs,numobs))/numobs).dot(self.response)
			return float(1 - ((numobs - 1)/(numobs - self.rank))*(SSE/SSTO))


			









class model_variable_editor(regression_model):

	def __init__(self,predictor_matrix,response):
		super().__init__(predictor_matrix,response)
	
	def __setitem__(self,column_number,transformation):
		"""
		Enter the column number as an integer. Note this does not include the bias column,
		whatever predictors you have entered for \'predictor_matrix\'.
		Accepted entries start from 0 up to number of predictors - 1, inclusive.
		You may also select a response column, but use negative integers.
		For example, if you're NOT doing multiple regression, use -1 for the only
		response column there is. Otherwise, use -2, -3, ... - (number of responses).
		Values greater than number of predictors - 1 are treated as new columns added to the
		matrix. Values equal to 0 or less than - (number of responses) are prohibited.
		Press \'q\' to quit.
		"""
		if not isinstance(column_number,int):
			raise Exception("Enter the column number as an integer. Pass \'yourinstancevariable.__setitem__\' into the \'help\' function for more info.")
		if not isinstance(transformation,str):
			raise Exception('Transformation must be a string.')

		if -1*self.num_responses <= column_number <= -1:
			self.changed = self.response[:,abs(column_number)-1]
			try:
				self.response[:,abs(column_number)-1] = getattr(np,transformation)(self.changed)
				super().__init__(self.X[:,1:],self.response)
				
			except NameError:
				raise Exception("Please import \'numpy\' as \'np\'. Transformation must belong to the \'Numpy\' library.")


			
		 



	# revert operation (return original state)

	# add more magic methods to compare >,< models based on AIC Cp Ra etc, 
	# select and modify & create and paste columns using self as well as transform __getitem__(key) or self[key]
	# add regularization and scikit learn api for higher polynomial regression
	# outlier detection via sckit-learn 

	# __repr__ or __str__ and display everything about the model

	# find a way through github to suggest edits and commit
 
		


#mc = model_variable_editor(data[:,:-1],np.abs(data[:,-1].reshape(-1,1)))
# print(mc.compute_beta())
# print(mc.response)
# mc[-1]='log'
# print(mc.compute_beta())
# print(mc.response)

mc = model_analysis(data[:,:-1],np.abs(data[:,-1].reshape(-1,1)),alpha=0.05)

#print(mc.confidence_interval_t_test_beta(num_comparisons=4,test_if_not_zero=True))
#print(mc.Beta_hat)



















