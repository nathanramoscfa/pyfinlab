
# PyFinanceLab

PyFinanceLab is a library which brings together various financial applications into one package for research and portfolio management. PyFinanceLab is in pre-alpha development. Please open an issue if you find any bugs. 



## Features

* **Data Api Wrapper**
    
    The data api wrapper makes it easy to switch between [yfinance](https://github.com/ranaroussi/yfinance) (free to use) and [tia](https://github.com/PaulMest/tia) (Bloomberg Professional Service subscription required) libraries for pulling financial data. 
    



## Installation

PyFinanceLab comes with many dependencies. It is recommended you use Anaconda for this installation. [Anaconda Individual Edition](https://www.anaconda.com/products/individual) is appropriate for most users. These instructions use Anaconda Prompt and pip for environment management and package installation. It is recommended you create a new Anaconda environment to keep pyfinlab isolated from your other environments. 

### Windows Instructions

Open Anaconda Prompt and create a new environment called pyfinlab. 
```
conda create -n pyfinlab python=3.8
```

Activate the new pyfinlab environment. 
```
conda activate pyfinlab
```

Install the following pip packages. 
```
pip install portfoliolab pyportfolioopt git+https://github.com/PaulMest/tia.git#egg=tia yfinance tqdm pyfinlab ffn bt
```

Install the following conda package. 
```
conda install -c conda-forge blpapi
```

Install JupyterLab. 
```
conda install -c conda-forge jupyterlab
```

Start the python interpreter and see if you can import the package modules. If the installation was successful, your python interpreter should look something like this. Please open an issue if you encounter any errors. 
```
python
>>> import portfoliolab, pypfopt, tia, yfinance, tqdm, pyfinlab, ffn, bt, blpapi
>>>
```



## Roadmap

Future development will include:

* **Classification Schema**

    Classify an investment universe of tickers into specified categories such as sector, size, or value. 

* **Constraints Modeling**

    Automatically generate weight constraints for a universe of tickers. 

* **Risk Modeling**

    Sample, test, and select the best risk model for generating covariance matrices for input into portfolio optimizers such as mean-variance optimization (MVO). Examples           include empirical covariance, ledoit-wolf shrinkage, minimum covariance determinant, and more.  

* **Portfolio Optimization**

    Utilize the classification schema, constraints modeling, risk modeling, and return modeling to optimize a portfolio of assets. 
    
* **Portfolio Backtesting**

    Backtest portfolios and generate performance graphical plots and statistics. 

* **Report Generation**

    Report results in a nicely formatted and easily readable Excel file. 
    
* **Documentation**

    Documentation will be published as this Python library is further developed. 

