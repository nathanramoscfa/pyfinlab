
# PyFinanceLab

PyFinanceLab is a library which brings together various financial applications into one package for research and portfolio management. PyFinanceLab is in pre-alpha development. Please open an issue if you find any bugs. 



## Features

* **Data Api Wrapper**
    
    The data api wrapper makes it easy to switch between [yfinance](https://github.com/ranaroussi/yfinance) (free to use) and [tia](https://github.com/PaulMest/tia) (Bloomberg Professional Service subscription required) libraries for pulling financial data. 
    



## Installation

PyFinanceLab comes with many dependencies. It is recommended you use Anaconda for this installation. [Anaconda Individual Edition](https://www.anaconda.com/products/individual) is appropriate for most users. These instructions use Anaconda Prompt and pip for environment management and package installation. It is recommended you create a new Anaconda environment to keep pyfinlab isolated from your other environments. 

Please open an issue if you have any problems. 

### Windows Instructions

Create a folder called "pyfinlab" wherever you normally store your Python project files. An example might be: 
```
C:\Users\User\Python Projects\pyfinlab
```
Open Anaconda Prompt and navigate to the folder you just created with the `cd` command.  
```
cd C:\Users\User\Python Projects\pyfinlab
```
Start python with the following command. 
```
python
```
Download the environment.yml file by copy and pasting the following code into the python console and press enter twice. 
```
import requests
r = requests.get('https://raw.githubusercontent.com/nathanramoscfa/pyfinlab/main/environment.yml')  
with open('environment.yml', 'wb') as f:
    f.write(r.content)
    exit()
```
Create a new Anaconda environment. 
```
conda env create -f environment.yml
```
Activate the new Anaconda environment. 
```
conda activate pyfinlab
```
Install the latest package version using pip. 
```
pip install pyfinlab
```
Check to see if you can import pyfinlab modules. 
```
python
>>> import pyfinlab
>>> 
```

#### Install JupyterLab 

Open Anaconda Navigator. In the "Applications on" menu located near the top, click the menu-arrow and select pyfinlab from the dropdown menu. Then find JupyterLab and click the "Install" button. If you already have Jupyter Lab installed, the button will say "Launch". You should now have everything you need to use pyfinlab and run Jupyter notebooks. 






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

