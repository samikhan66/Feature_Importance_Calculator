Feature Importance Calculator
========
This project helps calculate the feature importance of energy burden. In the US, there are several factors that influence the energy burden. In this project, I have come up with predictors and the relative role they play in determining how much energy burden an area (census tract) faces.

The software is used to train, test and select the best features necessary to predict the energy burden.

The main presentation is in the main.ipynb/main.html jupyter notebook and it will also have results for Colorado and Georgia notebooks in the last cell.

Additionally, I also packaged the assignment into an app for better modularization and readability that you can be run via instructions below:

# Installation
* cd to the project directory  
* Create virtual environment  
```
python3 -m venv /path/to/new/virtual/environment
```
* Activate virtual environment  
```
## for linux
source <your-virtual-environment>/bin/activate
## for windows
<your-virtual-environment>\Scripts\activate.bat
```
* cd into the project folder  
`cd Assignment`
* Upgrade pip and setuptools
```
pip install --upgrade pip
pip install --upgrade setuptools
```

* Install requirements  
```
## Install requirements
pip install -r requirements.txt
```

* Update options.ini in your text editor of choice  

* Run  
`python main.py`
