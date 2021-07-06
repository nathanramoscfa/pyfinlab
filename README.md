# PyFinanceLab

PyFinanceLab is in alpha development. Please open an issue if you find any bugs. 

## Features

**PyFinanceLab** is a library of financial applications focusing on portfolio management for Python. 

* **Data Api Wrapper**
    
    The data api wrapper makes it easy to switch between [yfinance](https://github.com/ranaroussi/yfinance) (free) and [tia](https://github.com/bpsmith/tia) (Bloomberg terminal     subscription required) libraries for pulling financial data. 
    

* **Covariance Matrix Tester**
    
    The covariance matrix tester tests various risk models from Hudson and Thames's [PortfolioLab](https://hudsonthames.org/portfoliolab/) Python library (requires subscription)     for computing covariance matrices out-of-sample on daily price data. Each risk model is ranked according to how close the covariance matrix generated with that risk             model is to out-of-sample empirical covariance matrix. 
    

## Roadmap

Future development will include:

* **Classification Schema**

    Classify an investment universe of tickers into specified categories such as sector, size, or value. 


* **Constraints Modeling**

    Automatically generate weight constraints for a universe of tickers. 


* **Risk Modeling**

    Sample, test, and select the best risk model for generating covariance matrices for input into portfolio optimizers such as mean-variance optimization (MVO). Examples           include empirical covariance, ledoit-wolf shrinkage, minimum covariance determinant, and more. 


* **Return Modeling**

    Sample, test, and select the best return model for generating return vectors for input into portfolio optimizers such as mean-variance optimization (MVO). Examples include       mean historical return, exponentially-weighted return, and the Capital Asset Pricing Model (CAPM). 


* **Portfolio Optimization**

    Utilize the classification schema, constraints modeling, risk modeling, and return modeling to optimize a portfolio of assets. 
    

* **Portfolio Backtesting**

    Backtest portfolios and generate performance graphical plots and statistics. 


* **Report Generation**

    Report results in a nicely formatted and easily readable Excel file. 
    

* **Documentation**

    Documentation will be published as this Python library is further developed. 
"# pyfinlab" 
