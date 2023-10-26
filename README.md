# AS-SEIRS
SEIRS Model source code
# Description
This model takes viral load in wastewater and clinical case data as input data for calibration of the model.
Estimated wastewater and Case are then compared with the measurment and corrected using Kalman filter.
# How to use
Use main-for-paper.ipynb for data analysis.
Use main-for-paper-scan.ipynb for scanning size of timing window for calibration
## Input data
Clinical case and virus concentration in wastewater
## Output data
Pdf of 
* effective reproduction number
* estimation of case data using only
* 1-week prediction 
# Notice
This program is a derivative of "Model-based assessment of COVID-19 epidemic dynamics by wastewater analysis" by Proverbio et al. (2022), which was under the Apache License 2.0. You can find the original source code at https://gitlab.lcsb.uni.lu/SCG/cowwan. This Program is also released under the Apache License 2.0.
## Modifications and Additions
In the process of translating the code from MATLAB to Python, several modifications and additions have been made. Here are some notable changes:
* Modification #1: Defined reinfection.
* Addition #1: Added cenario projections that increasing vaccination coverage and reducing contact rates.

Please note that this is an independent project and has not been endorsed by the original creators.