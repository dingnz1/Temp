""" A Silly Automatic Regression Modeler """

''' Use this program to find a regression model for two variables '''

import os
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

def get_valid_filename_y():
    ''' Ask for a filename until one is given '''
    # Input a txt file
    # There is only one value per line
    filename = input(
        'Input the file name including the dependent variable (y)? ')
    while os.path.isfile(filename) == False:
        print('File does not exist.')
        filename = input(
            'Input the file name including the dependent variable (y)? ')
    return filename

def get_valid_filename_x():
    ''' Ask for a filename until one is given '''
    # Same rule: txt file and one value per line
    filename = input(
        'Input the file name including the independent variable (x)? ')
    while os.path.isfile(filename) == False:
        print('File does not exist.')
        filename = input(
            'Input the file name including the independent variable (x)? ')
    return filename

def length_error():
    ''' End if the numbers of x and y values are not equal '''
    print('Conclusion:')
    print('The numbers of the two variables are not equal.')
    print('The dataset is not fit for regression.')

def read_records_from_file(filename):
    ''' Read data from the sample files '''
    value = []
    infile = open(filename)
    data = infile.readlines()
    for i in range(0, len(data)):
        temp = data[i].strip()
        value.append(temp)
    return value

def organize_data(value_x, value_y):
    ''' Sort the x values in an ascending order '''
    ''' Keep only the numeric values '''
    value = []
    type_value = 0
    for i in range(0, len(value_y)):
        # Filter the non-numeric values and the NAs out
        # Both x and y must be numeric so that this tuple (x, y) is kept
        temp = 0
        try:
            float(value_x[i])
            float(value_y[i])
            temp = 1
        except:
            ValueError
        if temp == 1:
            value.append((float(value_x[i]), float(value_y[i])))
    value = sorted(value)
    value_x = []
    for i in range(0, len(value)):
        value_x.append(value[i][0])
    value_y = []
    for i in range(0, len(value)):
        value_y.append(value[i][1])
    return value_x, value_y

def basic_statistics(value, variable_type=0):
    ''' Present the basic statistics '''
    mean = statistics.mean(value)
    median = statistics.median(value)
    std = statistics.stdev(value)
    print('---------------')
    if variable_type == 1:
        print('The mean of the dependent variable (y):',
              '{0:.2f}'.format(mean), '.')
        print('The median of the depenedent variable (y):',
              '{0:.2f}'.format(median), '.')
        print('The standard deviation of the dependent variable (y):',
              '{0:.2f}'.format(std), '.')
    if variable_type == 0:
        print('The mean of the independent variable (x):',
              '{0:.2f}'.format(mean), '.')
        print('The median of the indepenedent variable (x):',
              '{0:.2f}'.format(median), '.')
        print('The standard deviation of the independent variable (x):',
              '{0:.2f}'.format(std), '.')        

def plot(value_x, value_y):
    ''' Plot the points '''
    plt.plot(value_x, value_y, 'ro', color = 'black')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.show()

def linear_reg(value_x, value_y):
    ''' Linear regression '''
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    model = LinearRegression()
    model.fit(x, y)
    model = LinearRegression().fit(x, y)
    # Print the ouputs!
    print('---------------')
    print('Linear Regression')
    print('The intercept:', model.intercept_)
    print('The slope:', model.coef_)
    y_pred = model.predict(x)
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')    
    # Get the MSE
    # Later I recognized writing codes to calculate MSEs is not necessary
    mse = 0
    for i in range(0, len(y_pred)):
        mse += (value_y[i] - y_pred[i]) ** 2
        mse = mse / len(y_pred)
    print('The MSE:', mse)
    # Plot!
    plt.plot(x, y_pred, color = 'red')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.show()
    return mse

# Polynimial regression works with multiple degrees and it stable

# Polynimial regression works with multiple degrees and it stable

def poly_reg_2(value_x, value_y):
    ''' Polynomial regression (degree = 2) '''  
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    poly_reg = PolynomialFeatures(degree = 2)
    x_poly = poly_reg.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    # Print the ouputs!
    print('---------------')
    print('Polynomial Regression (Degree = 2)')
    print('The intercept:', model.intercept_)
    print('The coefficients:', model.coef_)
    y_pred = model.predict(poly_reg.fit_transform(x))
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')     
    # Get the MSE
    list_y_pred = list(y_pred)
    mse = 0
    for i in range(0, len(list_y_pred)):
        mse += (value_y[i] - list_y_pred[i]) ** 2
        mse = mse / len(list_y_pred)
    print('The MSE:', mse)
    # Plot!    
    plt.plot(x, y_pred, color = 'orange')
    plt.show()
    return mse

def poly_reg_4(value_x, value_y):
    ''' Polynomial regression (degree = 4) '''  
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    poly_reg = PolynomialFeatures(degree = 4)
    x_poly = poly_reg.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    # Print the ouputs!
    print('---------------')
    print('Polynomial Regression (Degree = 4)')
    print('The intercept:', model.intercept_)
    print('The coefficients:', model.coef_)
    y_pred = model.predict(poly_reg.fit_transform(x))
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')  
    # Get the MSE
    list_y_pred = list(y_pred)
    mse = 0
    for i in range(0, len(list_y_pred)):
        mse += (value_y[i] - list_y_pred[i]) ** 2
        mse = mse / len(list_y_pred)
    print('The MSE:', mse)
    # Plot!    
    plt.plot(x, y_pred, color = 'yellow')
    plt.show()
    return mse

def poly_reg_8(value_x, value_y):
    ''' Polynomial regression (degree = 8) '''  
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    poly_reg = PolynomialFeatures(degree = 8)
    x_poly = poly_reg.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    # Print the ouputs!
    print('---------------')
    print('Polynomial Regression (Degree = 8)')
    print('The intercept:', model.intercept_)
    print('The coefficients:', model.coef_)
    y_pred = model.predict(poly_reg.fit_transform(x))
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')  
    # Get the MSE
    list_y_pred = list(y_pred)
    mse = 0
    for i in range(0, len(list_y_pred)):
        mse += (value_y[i] - list_y_pred[i]) ** 2
        mse = mse / len(list_y_pred)
    print('The MSE:', mse)
    # Plot!    
    plt.plot(x, y_pred, color = 'green')
    plt.show()
    return mse

def poly_reg_16(value_x, value_y):
    ''' Polynomial regression (degree = 16) '''  
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    poly_reg = PolynomialFeatures(degree = 16)
    x_poly = poly_reg.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    # Print the ouputs!
    print('---------------')
    print('Polynomial Regression (Degree = 16)')
    print('The intercept:', model.intercept_)
    print('The coefficients:', model.coef_)
    y_pred = model.predict(poly_reg.fit_transform(x))
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')  
    # Get the MSE
    list_y_pred = list(y_pred)
    mse = 0
    for i in range(0, len(list_y_pred)):
        mse += (value_y[i] - list_y_pred[i]) ** 2
        mse = mse / len(list_y_pred)
    print('The MSE:', mse)
    # Plot!    
    plt.plot(x, y_pred, color = 'cyan')
    plt.show()
    return mse

# Not every regression works well in a program that is for a general purpose
# For example, logistic regression is meaningless for a large number of outcomes
# In other words, it works the best when y values include only a and b

def func_log(x, a, b, c):
    ''' Calculate logs '''
    return a * np.log(b * x) + c

def logarithmic_reg(value_x, value_y):
    ''' Logarithmic regression '''
    x = np.array(value_x)
    y = np.array(value_y)
    # Get the best parameters
    popt, pcov = curve_fit(func_log, x, y, p0 = (-100, 0.01, 100))
    # Print the ouputs!
    print('---------------')
    print('Logarithmic Regression')
    print('The parameters:', popt)
    y_pred = func_log(x, *popt)
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')  
    # Get the MSE
    list_y_pred = list(y_pred)
    mse = 0
    for i in range(0, len(list_y_pred)):
        mse += (value_y[i] - list_y_pred[i]) ** 2
        mse = mse / len(list_y_pred)
    print('The MSE:', mse)
    # Plot!
    if mse > 0 and mse < 1:
        plt.plot(x, y_pred, color = 'blue')
        plt.show()
    else:
        plt.plot(0, 0, color = 'blue')
    return mse

def ridge_reg(value_x, value_y):
    ''' Ridge regression '''  
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    # Get the best value of alpha
    alphas = np.linspace(.00001, 2, 500)
    ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error',
                      normalize = True)
    ridgecv.fit(x, y)    
    model = Ridge(alpha = ridgecv.alpha_)
    model.fit(x, y)
    print('---------------')
    print('Ridge Regression')
    print('The alpha parameter:', ridgecv.alpha_)
    y_pred = model.predict(x)
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + 
          ' responses are omitted.)')    
    # Get the MSE
    mse = 0
    for i in range(0, len(y_pred)):
        mse += (value_y[i] - y_pred[i]) ** 2
        mse = mse / len(y_pred)
    print('The MSE:', mse)
    # Plot!
    plt.plot(x, y_pred + (statistics.mean(y) * 0.01), color = 'indigo')
    # '+ (statistics.mean(y) * 0.01)' is to prevent the overlapping
    plt.show()
    return mse

def lasso_reg(value_x, value_y):
    ''' Lasso regression '''  
    x = np.array(value_x).reshape((-1, 1))
    y = np.array(value_y)
    lasso = Lasso()
    # Get the best value of alpha
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    model = GridSearchCV(lasso, parameters,
                         scoring = 'neg_mean_squared_error', cv = 5)
    model.fit(x, y)
    print('---------------')
    print('Lasso Regression')
    print('The alpha parameter:', model.best_params_)
    print('The score:', model.best_score_)
    y_pred = model.predict(x)
    print('The predicted responses:', y_pred[: 20], sep = '\n')
    if len(value_y) > 20:
        print('(Only the first 20 responses are printed. ' +
              str((len(value_y) - 20)) + ' responses are omitted.)')    
    # Get the MSE
    mse = 0
    for i in range(0, len(y_pred)):
        mse += (value_y[i] - y_pred[i]) ** 2
        mse = mse / len(y_pred)
    print('The MSE:', mse)
    # Plot!
    plt.plot(x, y_pred + (statistics.mean(y) * 0.02), color = 'violet')
    plt.show()
    return mse

def add_legend():
    ''' Add legends to the plot '''
    plt.legend(loc = 'best', labels = ['Observation', 'Linear Rregreesion',
                                       'Polynomial Regression (Degree = 2)',
                                       'Polynomial Regression (Degree = 4)',
                                       'Polynomial Regression (Degree = 8)',
                                       'Polynomial Regression (Degree = 16)',
                                       'Logarithmic Regression',
                                       'Ridge Regression',
                                       'Lasso Regression'])

def recommend_model(models):
    ''' Recommend the model with the smallest MSE '''
    best_mse = float('inf')
    for i in range(0, len(models)):
        if models[i][0] < best_mse:
            best_mse = models[i][0]
            best_model = models[i][1]
    return str('{:.5f}'.format(best_mse)), best_model

def print_final(best_mse, best_model, filename_y, filename_x):
    ''' Print the final ouputs '''
    print('---------------')
    print('---------------')
    print('Conclusion:')
    if float(best_mse) <= 0.25:
        print('The best regression model for the two variables '
              + filename_y[: -4]
              + ' (y) and '
              + filename_x[: -4]
              + ' (x) is the '
              + best_model
              + ' model, with the smallest mean squared error '
              + best_mse
              + '.')
    else:
        print('No regression model in this program is recommended for this dataset.')    

def main():
    ''' Main program '''
    warnings.filterwarnings('ignore')    
    print('------------------------------')
    # Preliminary processing
    filename_y = get_valid_filename_y()
    filename_x = get_valid_filename_x()
    value_y = read_records_from_file(filename_y)
    value_x = read_records_from_file(filename_x)
    if len(value_y) != len(value_x):
        length_error()
    else:
        value_x, value_y = organize_data(value_x, value_y)
        # Basic statistics
        basic_statistics(value_y, 1)
        basic_statistics(value_x, 0)
        # Plots
        plot(value_x, value_y)
        # Linear regression
        mse_linear = linear_reg(value_x, value_y)
        # Polynomial regression with multiple degress
        mse_poly_2 = poly_reg_2(value_x, value_y)
        mse_poly_4 = poly_reg_4(value_x, value_y)
        mse_poly_8 = poly_reg_8(value_x, value_y)
        mse_poly_16 = poly_reg_16(value_x, value_y)
        # Logarithmic regression
        mse_log = logarithmic_reg(value_x, value_y)
        # Ridge regression
        mse_ridge = ridge_reg(value_x, value_y)
        # Lasso regression
        mse_lasso = lasso_reg(value_x, value_y)
        add_legend()
        # Print the final ouputs!
        models = [(mse_linear, 'linear regression'), 
                  (mse_poly_2, 'polynomial regression (degree = 2)'), 
                  (mse_poly_4, 'polynomial regression (degree = 4)'), 
                  (mse_poly_8, 'polynomial regression (degree = 8)'),
                  (mse_poly_16, 'polynomial regression (degree = 16)'),
                  (mse_log, 'logarithmic regression'),
                  (mse_ridge, 'Ridge regression'),
                  (mse_lasso, 'Lasso regression')]
        best_mse, best_model = recommend_model(models)
        print_final(best_mse, best_model, filename_y, filename_x)
    print('------------------------------')
    
main()