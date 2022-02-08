import matplotlib.pyplot as plt
import numpy as np
from tqdm import tnrange
import scipy.stats
import pandas as pd
from scipy.stats import rankdata
import matplotlib as mpl
import itertools

class WeightedCorr:
    def __init__(self, xyw=None, x=None, y=None, w=None, df=None, wcol=None):
        ''' Weighted Correlation class. Either supply xyw, (x, y, w), or (df, wcol). Call the class to get the result, i.e.:
        WeightedCorr(xyw=mydata[[x, y, w]])(method='pearson')
        :param xyw: pd.DataFrame with shape(n, 3) containing x, y, and w columns (column names irrelevant)
        :param x: pd.Series (n, ) containing values for x
        :param y: pd.Series (n, ) containing values for y
        :param w: pd.Series (n, ) containing weights
        :param df: pd.Dataframe (n, m+1) containing m phenotypes and a weight column
        :param wcol: str column of the weight column in the dataframe passed to the df argument.
        '''
        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError('No data supplied')
            if not ((isinstance(xyw, pd.DataFrame)) != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))):
                raise TypeError('xyw should be a pd.DataFrame, or x, y, w should be pd.Series')
            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (pd.to_numeric(xyw[i], errors='coerce').values for i in xyw.columns)
            self.df = None
        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError('df should be a pd.DataFrame and wcol should be a string')
            if wcol not in df.columns:
                raise KeyError('wcol not found in column names of df')
            self.df = df.loc[:, [x for x in df.columns if x != wcol]]
            self.w = pd.to_numeric(df.loc[:, wcol], errors='coerce')
        else:
            raise ValueError('Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)')

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])
        return self._wcov(x, y, [mx, my]) / np.sqrt(self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my]))

    def _wrank(self, x):
        #(unique, arr_inv, counts) = np.unique(rankdata(x), return_counts=True, return_inverse=True)
        #a = np.bincount(arr_inv, self.w)
        #a = np.bincount(arr_inv, np.ones_like(self.w))
        #return (np.cumsum(a) - a)[arr_inv]+((counts + 1)/2 * (a/counts))[arr_inv]
        return rankdata(x)

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))
    
    def _mut_info(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        n_00_idx = np.where((self.x == 0) & (self.y == 0))[0]
        n_01_idx = np.where((self.x == 0) & (self.y == 1))[0]
        n_10_idx = np.where((self.x == 1) & (self.y == 0))[0]
        n_11_idx = np.where((self.x == 1) & (self.y == 1))[0]
        
        if n_00_idx.size == 0:
            n_00 = 0
        else:
            n_00 = np.sum(self.w[n_00_idx])
        if n_01_idx.size == 0:
            n_01 = 0
        else:
            n_01 = np.sum(self.w[n_01_idx])
        if n_10_idx.size == 0:
            n_10 = 0
        else:
            n_10 = np.sum(self.w[n_10_idx])
        if n_11_idx.size == 0:
            n_11 = 0
        else:
            n_11 = np.sum(self.w[n_11_idx])

        n_sum = n_00 + n_01 + n_10 + n_11
        
        n_00 /= n_sum
        n_01 /= n_sum
        n_10 /= n_sum
        n_11 /= n_sum
        
        if n_00 != 0:
            mi_00 = n_00 * (np.log(n_00) - np.log(n_00 + n_10) - np.log(n_00 + n_01))
        else:
            mi_00 = 0
        if n_01 != 0:
            mi_01 = n_01 * (np.log(n_01) - np.log(n_01 + n_11) - np.log(n_01 + n_00))
        else:
            mi_01 = 0  
        if n_10 != 0:
            mi_10 = n_10 * (np.log(n_10) - np.log(n_10 + n_00) - np.log(n_10 + n_11))
        else:
            mi_10 = 0
        if n_11 != 0:
            mi_11 = n_11 * (np.log(n_11) - np.log(n_11 + n_01) - np.log(n_11 + n_10))
        else:
            mi_11 = 0
        
        print(n_00 + n_01 + n_10 + n_11)
        
        return mi_00 + mi_01 + mi_10 + mi_11

    def __call__(self, method='pearson'):
        '''
        :param method: Correlation method to be used: 'pearson' for pearson r, 'spearman' for spearman rank-order correlation.
        :return: if xyw, or (x, y, w) were passed to __init__ returns the correlation value (float).
                 if (df, wcol) were passed to __init__ returns a pd.DataFrame (m, m), the correlation matrix.
        '''
        if method not in ['pearson', 'spearman', 'mut_info']:
            raise ValueError('method should be one of [\'pearson\', \'spearman\']')
        cor = {'pearson': self._pearson, 'spearman': self._spearman, 'mut_info': self._mut_info}[method]
        if self.df is None:
            return cor()
        else:
            out = pd.DataFrame(np.nan, index=self.df.columns, columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        out.loc[x, y] = cor(x=pd.to_numeric(self.df[x], errors='coerce'), y=pd.to_numeric(self.df[y], errors='coerce'))
                        out.loc[y, x] = out.loc[x, y]
            return out