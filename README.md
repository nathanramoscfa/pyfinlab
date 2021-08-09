


# PyFinanceLab

PyFinanceLab is a library which brings together various financial applications into one package for research and portfolio management. Navigate to the jupyter folder of the pyfinlab repository to see usage examples. 

PyFinanceLab is in pre-alpha development. Please open an issue if you find any bugs. 


## Features

* **Data API Wrapper**
	 The data API wrapper makes it easy to switch between [yfinance](https://github.com/ranaroussi/yfinance) (free to use) and [tia](https://github.com/PaulMest/tia) (Bloomberg Professional Service subscription required) libraries for pulling financial data. 

* **Portfolio Optimizer**
	Compute an efficient frontier of portfolios based on any one of 16 risk models and 6 return models from Hudson & Thame's [PortfolioLab](https://hudsonthames.org/portfoliolab/) or [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/) libraries. 

* **Optimizer Backtest**
	Backtest optimized portfolios and compute performance charts, efficient frontier plots, and performance statistics. 

* **Excel Report Generation**
	Show your optimizer results and backtest in a nicely formatted Excel file for further analysis. 
    

## Installation

PyFinanceLab comes with many dependencies. It is recommended you use Anaconda for this installation process. [Anaconda Individual Edition](https://www.anaconda.com/products/individual) is appropriate for most users. Make sure you have installed [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed on your computer. If you encounter any errors with, "Microsoft Visual C++ 14.0 is required", try following [these instructions](https://stackoverflow.com/a/55370133/16367225) to download and install Microsoft Visual C++ 14.0. Open an issue if you need help. 


### Windows Instructions

Open Anaconda Prompt and create a new environment called pyfinlab. 
```
conda create -n pyfinlab python=3.8 git
```

Activate the new pyfinlab environment. 
```
conda activate pyfinlab
```

Install the following pip packages. 
```
pip install portfoliolab yfinance tqdm pyfinlab openpyxl ffn patsy openpyxl
```

Install the following GitHub repository. 
```
pip install git+https://github.com/PaulMest/tia.git#egg=tia
```

Install the following conda packages using conda-forge channel. 
```
conda install -c conda-forge blpapi jupyterlab xlsxwriter tqdm
```

Install the following conda packages using anaconda channel. 
```
conda install -c anaconda xlsxwriter statsmodels
```
If you get an error, please open an issue. 


## Roadmap

Future development will include:

* **Multifactor Scoring Model**

    Analyze and rank every stock and ETF according to factors assumed to have excess returns and violate the efficient market hypothesis. 
    
* **Documentation**

    Documentation will be published as this Python library is further developed. 

