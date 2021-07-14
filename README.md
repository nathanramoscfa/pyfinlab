
# PyFinanceLab

PyFinanceLab is a library which brings together various financial applications into one package for research and portfolio management. PyFinanceLab is in pre-alpha development. Please open an issue if you find any bugs. 


## Features

* **Data Api Wrapper**
    
    The data api wrapper makes it easy to switch between [yfinance](https://github.com/ranaroussi/yfinance) (free to use) and [tia](https://github.com/PaulMest/tia) (Bloomberg Professional Service subscription required) libraries for pulling financial data. 
    

## Installation

PyFinanceLab comes with many dependencies. It is recommended you use Anaconda for this installation process. [Anaconda Individual Edition](https://www.anaconda.com/products/individual) is appropriate for most users. Make sure you have installed [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed on your computer. If you encounter any errors with, "Microsoft Visual C++ 14.0 is required", try following [these instructions](https://stackoverflow.com/a/55370133/16367225) to download and install Microsoft Visual C++ 14.0. Open an issue if you need help. 


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
pip install portfoliolab git+https://github.com/PaulMest/tia.git#egg=tia yfinance tqdm pyfinlab
```

Install the following conda packages. 
```
conda install -c conda-forge blpapi jupyterlab
```
Check to see if you can import pyfinlab modules. Your python interpreter should look like the following if the modules were successfully installed. If you get an error, please open an issue. 
```
python
>>> import portfoliolab, tia, blpapi, yfinance, tqdm, pyfinlab
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

