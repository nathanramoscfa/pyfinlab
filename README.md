# PyFinanceLab

PyFinanceLab is a project which brings together various financial applications into one package for research and portfolio management. PyFinanceLab is in pre-alpha development. Please open an issue if you find any bugs. 



## Features

**PyFinanceLab** is a library of financial applications focusing on portfolio management for Python. 

* **Data Api Wrapper**
    
    The data api wrapper makes it easy to switch between [yfinance](https://github.com/ranaroussi/yfinance) (free to use) and [tia](https://github.com/PaulMest/tia) (Bloomberg Professional Service subscription required) libraries for pulling financial data. 
    



## Installation Instructions

Open Anaconda Prompt and create a new environment using the environment.yml file. 

`conda env create -f environment.yml`

Activate new environment. 

`conda activate pyfinlab`

Install the latest version using pip. 

`pip install pyfinlab`




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

