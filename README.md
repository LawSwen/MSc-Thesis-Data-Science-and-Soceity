## Sparse-Aware Deep Learning for Air Quality Forecasting


## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> Project Overview </span>
This respository hosts the code for the MSc Thesis *Sparse-Aware Deep Learning for Air Quality Forecasting: Simulating Data Gaps to Enchance Model Robustness in Resource-Constrained Environments*\
This project investigates the stability and behavior of Seq2Seq model in the presence of sparsity. It follows with an alternative framwork that seeks to directly model sparsity rather then rely on imputaton methods. This frameworks is build with an encoder-decoder layers that uses Time-aware GRU with masked attetion mechanism to learn directly from sparsity in air qaulity series. When proven viable, this framework may be useful for places with limited sensor coverage, poor data standards and limited technical resources. 

## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> Repository Design </span>

<ul>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 20px;">EDA</li>
  <ul>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 18px"> Exploratory Data Analysis </li>
  </ul>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 20px;">models </li>
   <ul>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 18px"> Attention_model</li>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 18px"> SMA </li>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 18px"> SEq2Seq model</li>
  </ul>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 20px;">Utilities </li>
  <ul>
  <li style="color:white; text-decoration:underline; text-style:bold; font-size: 18px">data preparation </li>
    <li style="color:white; text-decoration:underline; text-style:bold; font-size: 18px"> requirements</li>
  </ul>
</ul> 

## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> EDA </span> 
The EDA branch contains the file <span style="background-color: #060270;">EDA.ipynb</span> This file runs the outlier analysis and the exploratory data analysis

## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> models </span> 
The model branch contains the files <span style="background-color: #060270;"> attention_model.py </span>,  <span style="background-color: #060270;">attention_model_plot.py </span>, <span style="background-color: #060270;">attention_model_optimization.ipynb </span> These three files run the attention model, plot it and also optimizes it... This model also have the Seq2Seq files for both the random sparsity and the imputation models.

## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> Utilities </span> 

The utilities branch contains the files <span style="background-color: #060270;"> attention_prepare.py </span>,  <span style="background-color: #060270;">Seq2Seq_prepare.py</span>, and  <span style="background-color: #060270;">all_imports.py </span> Each of these files host codes that are vital to preparing the data for each of the models. However the file <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> all_imports.py </span> must be in the same directory as the rest of the code for any of these models or codes to run without errors. This file contains all the loaded packages needed to run this entire pipeline.. 


## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> Files/Folder Structures </span> 
```
project/
├── EDA/
│   ├── Explotory_data_analysis/
|   |   |── EDA.ipynb
├── models/
│   ├── attention_models/
|   |   |── attention_model.py
|   |   |── attention_model_plot.py
|   |   |── attention_model_optimization.ipynb
│   └── Seq2Seq_models/
|   |   |── Seq2Seq_models_baseline.py
|   |   |── Seq2Seq_models_optimization.ipynb
|   |   |── imputation_pipeline.ipynb
|   |   |── random_sparsity_models.ipynb
├── utilities/
│   ├── data_preparation/
|   |   |── Seq2Seq_data_prep/
|   |   |   |── Seq2Seq_data_prep.py
|   |   |── attention_data_prep/
|   |   |   |── attetion_prepare.py
│   ├── requirements/
|   |   |   |── all_imports_file/
|   |   |   |     |── all_imports.py
|___
```
## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> Data Source </span> 
<p>
The air quality data used in this project was acquired from the 
<a href="https://www.eea.europa.eu/en" style="color: #0366d6; text-decoration: none; font-weight: bold; font-size: 20px">European Environmental Agency</a> 
and publicly available via the 
<a href="https://eeadmz1-downloads-webapp.azurewebsites.net/" style="color: #0366d6; text-decoration: none; font-weight: bold; font-size:20px">European Air Quality Download Service</a>.
</p>

## <span style="color:#DFC57B; text-decoration:underline; text-style:bold"> Ethics and Declarations </span> 
The owners of the data sets retain full ownership. All figures, codes and tables in this thesis are works of the author. For research and other scientific purposes, any parts of this work can be used or further extended as long the accompanying citations are stated.

<div align="center">
  <br>
  <p><strong>Author: Lawrence Swen Tuah</strong></p>
</div>