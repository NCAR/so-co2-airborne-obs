import numpy as np

from scipy import stats
import scipy.odr as odr


class linreg_odr(object):
    """Wrapper to SciPy's Orthogonal Distance Regression Package.
    The wrapping provides some ready access to goodness of fit 
    statistics as well as some helpful additional properties. 
    
    Parameters
    ----------
    
    x : array_like, optional
      Independent variable for regression. 
    
    y : array_like, optional
      Dependent variable.
    
    xerr : array_like
      Error estimate on values in `x`.
    
    yerr : array_like
      Error estimate on values in `y`.
    
    """

    
    def __init__(self, x=None, y=None, xerr=None, yerr=None):
        """Build the scipy.odr object, perform regression, set some properties.
        """
        
        self.odr = None
        self.n = 0        
        
        # if there's no data, 
        if x is None and y is None:
            return
        assert (x is not None) and (y is not None), "Require both x and y."
        assert ~np.any(np.isnan(x) | np.isnan(y)), "No nans allowed."
        
        self.n = len(x)
        self.data = odr.Data(x, y, wd=xerr, we=yerr)         
        self.odr = odr.ODR(self.data, odr.unilinear).run()
        self.xhat = np.sort(x)
        self.yhat = self.predict(self.xhat)

    @property
    def beta(self):
        """Estimated parameter values.
        """
        if self.odr is None:
            return np.ones((2,)) * np.nan        
        return self.odr.beta
    
    @property
    def res_var(self):
        """Residual variance"""
        if self.odr is None:
            return np.nan            
        return self.odr.res_var
        
    @property
    def r2(self):
        if self.odr is None:
            return np.nan        
        return self._calc_r2()

    @property
    def rmse(self):
        """Return the root mean square error of the fit in y.
        """
        if self.odr is None:
            return np.nan            
        return self._calc_rmse_y()
    
    @property
    def pval(self):
        """Return the p-value
        See page 76 of ODRPACK documentation available here:
        https://docs.scipy.org/doc/external/odrpack_guide.pdf
        """
        if self.odr is None:
            return np.nan            
        return self._calc_pval()
    
    @property
    def sse_y(self):
        if self.odr is None:
            return np.nan               
        return self._calc_sse_y()
        
    @property
    def cov_beta(self):
        if self.odr is None:
            return np.ones((2, 2)) * np.nan
        return self.odr.cov_beta
    
    @property
    def stderr_beta(self):
        if self.odr is None:
            return np.ones((2,)) * np.nan               
        # sd_beta = sqrt(diag(cov_beta * res_var))
        return self.odr.sd_beta #np.sqrt(np.diag(self.cov_beta))

    @property
    def s2n(self):
        if self.odr is None:
            return np.nan        
        return np.abs(self.beta[0]) / (self.cov_beta[0, 0]**0.5)
        
    def predict(self, xp):
        return self.beta[0] * xp + self.beta[1]

    def _calc_rmse_y(self):
        return np.sqrt(self._calc_sse_y() / self.n)

    def _calc_pval(self):
        """Compute p value of slope. 
        """
        t = self.beta / self.stderr_beta
        return (2. * (1. - stats.t.cdf(np.abs(t), self.n - 2)))[0]

    def _calc_sse_y(self):
        return np.sum((self.data.y - self.predict(self.data.x))**2)
    
    def _calc_r2(self):
        """Compute coefficient of determination.
        """
        sse = self._calc_sse_y()
        sst = np.sum((self.data.y - self.data.y.mean())**2)
        return (1. - sse/sst)  
    
    @property
    def persist_keys(self):
        """Set the properties that should be saved to document the results of the regression 
        analysis.
        """
        return [
            'beta',
            'stderr_beta',
            'cov_beta',
            'res_var',
            'r2',
            'rmse',
            'pval',
            'sse_y',
            's2n',
            'n',
        ]
    
    def to_dict(self):        
        """Return a dictionary documenting the results of the regression analysis.        
        """
        return {k: self.__dict__[k] for k in self.persist_keys}        
       