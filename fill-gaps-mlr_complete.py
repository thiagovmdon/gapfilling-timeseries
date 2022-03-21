# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:51:03 2022

@author: Thiago Nascimento
Code developed for filling missing data in time-series using multiple linear regression (MLR)
In this code we take into consideration the t-statistic value >= 2
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
import datetime
import warnings
from scipy.spatial.distance import  cdist
from sklearn import linear_model

warnings.simplefilter(action='ignore', category=Warning)
plt.close("all")

# Set and print your working directory
os.chdir(r'C:\Users\User\OneDrive\ERASMUS\4_Thesis\python\MLR')
print("Current Working Directory " , os.getcwd())

# Read the xlsx data with the coordinates (x,y) of your points
pathcoords =r'C:\Users\User\OneDrive\ERASMUS\4_Thesis\python\MLR\raing_gauges_unfilled_coords.xlsx'
coords=pd.read_excel(pathcoords)
coords.rename(columns={"CÓDIGO": "Code"}, inplace=True)
coords.set_index('Code', inplace=True)

# Read the xlsx file with your nonfilled data
pathnonfilleddata =r'C:\Users\User\OneDrive\ERASMUS\4_Thesis\python\MLR\raing_gauges_unfilled.xlsx'
nonfilleddata=pd.read_excel(pathnonfilleddata)
nonfilleddata.set_index('dates', inplace=True)

num_postos = nonfilleddata.shape[1] #Number of points
num_dias = nonfilleddata.shape[0] #Total time lenght 

# For the case that you are working with a sub-set of your main dataset, you can define a filter
# with the points that will be corrected. This reduces computation time when you are not interested
# on filling all the points:
# If you are not working with a filter, please select 0:

work_w_filter = input(" Are you working with a subset of your data? (Yes or Not). And if yes, provide a filter xlsx: ")

if work_w_filter == "Yes":
    pathfilter =r'C:\Users\User\OneDrive\ERASMUS\4_Thesis\python\MLR\filter.xlsx'
    filtaux=pd.read_excel(pathfilter)
    id_filter = filtaux["CÓDIGO"]    
    num_corrigir = len(id_filter)
    nonfilleddata_corrigir = nonfilleddata[id_filter]
    print("The number of points to be corrected is: " + str(num_corrigir))
else:
    num_corrigir = num_postos
    nonfilleddata_corrigir = nonfilleddata


# Calculate the percentage of failures per point:
desc = nonfilleddata.describe()
perc_erros = pd.DataFrame(index = coords.index)
perc_erros["perc_erros"] = (1 - desc.iloc[0,:]/num_dias)*100

#Create a distance matrix:
#Convert the coords to a numpy matrix
coords_np = coords[["Lon", "Lat"]].to_numpy()
dist_mat = cdist(coords_np, coords_np,metric = 'euclidean')
#Convert again for a dataframe
dist_mat_df=pd.DataFrame(dist_mat)
dist_mat_df.columns=coords.index
dist_mat_df.index=coords.index

#Dataframe that will be filled
filleddata = nonfilleddata_corrigir

#%% 
# The loop will work for each point to be corrected
# At this point you can choose if you take into consideration the distance as a limitant factor, or not
# In addition, you can choose to use or not the t-statistic as also a limitant

Use_distance = input("Do you want to take into consideration the distance between points? (Yes/ No) ")
Use_t = input("Do you want to take into consideration the t-statistic of your MLR, i.e., only points with t-stat >= 2 will be considered? (Yes/ No) ")


#Maximum number of gauges used for correction. This only makes sense if you want to use the distance between the wells
#as a limitant for your MLR, i.e., if you want to use only the n-closest wells for you MLR. 
num_correcao = 100000

for i in range(num_corrigir):
#for i in range(1):
    name = nonfilleddata_corrigir.columns[i] #Point's name
    index = nonfilleddata_corrigir[name].index[nonfilleddata_corrigir[name].apply(np.isnan)] #Indexes of the point with gaps
    
    #This loop will correct each day with gap in the point
    for j in index:
        
        #Pay attention that only points with no failures at the day to be corrected in the point to be corrected can be used for the model creation and regression
        names_0_that_day = nonfilleddata.columns[nonfilleddata.loc[j].apply(np.isfinite)] #Code of the points with no errors at this day
        num_ava_points = len(names_0_that_day)
        
        # If there are no points with zero failures at that day, the filling cannot take place
        if num_ava_points == 0:
            filleddata[name].loc[j] = np.nan
        
        else:
            if Use_distance == "Yes":

                # Depending on the num_correcao that you defined, it is possible that the number of available points to be used for corrrection is smaller than this maximum number, therefore 
                # the code will have to use only the available points
                if num_ava_points >= num_correcao:
                    nclosest = dist_mat_df[names_0_that_day].loc[name].nsmallest(num_correcao)
                else:
                    nclosest = dist_mat_df[names_0_that_day].loc[name].nsmallest(num_ava_points)
        
                # Name of the closest points to be used for correction
                names_closest = nclosest.index
            else:
                names_closest = names_0_that_day             
                
             
            # Matrix within the [X y] format
            datamatrix = nonfilleddata[names_closest].join(nonfilleddata[name])
            # Rows with NaN in either of our matrix cannot be used for regression, therefore we must drop them
            datamatrix.dropna(inplace = True)
            
            # The len of the matrix is calculated
            len_mat = len(datamatrix)
            
            # If the len of the matrix is lower than 0 (that can happen), it is not possible to fill the gap
            if len_mat <= 1:
                filleddata[name].loc[j] = np.nan
            
            # Else, the calculation can proceed
            else:
                # Save the y and X
                y = datamatrix.iloc[:,-1]
                X = datamatrix.iloc[:,:-1]
                # If the y column is formed only with 0s, it is not possible to proceed
                if (y != 0).any(axis=0):
                    
                    # In addition, columns (names-X) that have only 0 as measurements are as well deleted
                    X = X.loc[:, (X != 0).any(axis=0)]
                    names_closest = X.columns #And the names of the ones used for correction are also updated
                    
                    # If the len of the matrix is lower than 0 (that can happen), it is not possible to fill the gap
                    if len(names_closest) < 1:
                        filleddata[name].loc[j] = np.nan
                    else:
                        #%%Multiple linear regression
                        regr = linear_model.LinearRegression()
                        regr.fit(X, y)
                        
                        #Maybe the matrix will remain as singular even without the 0 columns, thus, we have to test, and if this is the case, we can proceed the calculation do not taking into consideration the |t-stats| <=2
                        newX = pd.DataFrame({"Constant":np.ones(len(X))}, index = X.index).join(X)
                        if Use_t != "Yes" or np.linalg.det(np.dot(newX.T,newX)) ==0:
                            # Gaps filling
                            filleddata[name].loc[j] = regr.predict([nonfilleddata[names_closest].loc[j]])
                    
                        else:
                            #t-statistics
                            params = np.append(regr.intercept_,regr.coef_)
                            predictions = regr.predict(X)
                
                
                            newX = pd.DataFrame({"Constant":np.ones(len(X))}, index = X.index).join(X)
                            MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

                            var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
                            sd_b = np.sqrt(var_b)
                            ts_b = params/ sd_b

                
                            myDF3 = pd.DataFrame()
                            myDF3["Coefficients"],myDF3["t values"]= [params,ts_b]
                
                
                            myDF4 = myDF3[1:] 
                            myDF4 = myDF4.set_index(names_closest)
    
                            filt = ((myDF4['t values'].abs() >= 2))
                            new_names = myDF4.loc[filt].index
                
                            # We tested if there are points with abs(t-stats) < 2, and if yes, we re-calculate MLR:
                            if len(new_names) == len(names_closest):
                                # Gaps filling
                                filleddata[name].loc[j] = regr.predict([nonfilleddata[names_closest].loc[j]])
                
                            else:
                                # Matrix within the [X y] format
                                datamatrix_new = nonfilleddata[new_names].join(nonfilleddata[name])
                                # Rows with NaN in either of our matrix cannot be used for regression, therefore we must drop them
                                datamatrix_new.dropna(inplace = True)
                 
                                # Save the y and X
                                y_new = datamatrix_new.iloc[:,-1]
                                X_new = datamatrix_new.iloc[:,:-1]
                                # If the y column is formed only with 0s, it is not possible to proceed
                                if (y_new != 0).any(axis=0):
                 
                                    # In addition, columns (names-X) that have only 0 as measurements are as well deleted
                                    X_new = X_new.loc[:, (X_new != 0).any(axis=0)]
                                    new_names = X_new.columns #And the names of the ones used for correction are also updated
                                    # If the len of the matrix is lower than 0 (that can happen), it is not possible to fill the gap
                                    if len(new_names) < 1:
                                        filleddata[name].loc[j] = np.nan
                                    else:
                                        #%%Multiple linear regression
                                        regr_new = linear_model.LinearRegression()
                                        regr_new.fit(X_new, y_new)
             
                                        # Gaps filling
                                        filleddata[name].loc[j] = regr_new.predict([nonfilleddata[new_names].loc[j]])   
                                else:
                                    filleddata[name].loc[j] = np.nan 
                else:
                    filleddata[name].loc[j] = np.nan             
                       

# It is possible that negative values will be calculated, thus we replace them per 0:
filleddata[filleddata < 0.1] = 0
    


















