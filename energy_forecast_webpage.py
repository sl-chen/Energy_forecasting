# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:53:21 2019

@author: Shanlin Chen
"""
#import modules and functions
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #use external stylesheets 

app = dash.Dash(__name__, external_stylesheets=external_stylesheets) #build app by using dash

df = pd.read_csv('ISTPower_2018_19.csv', index_col=0) # read csv.file
df.index = pd.to_datetime(df.index) # change the index to datetime

app.layout = html.Div(children=[
    html.H1(children='IST Energy Monitor',
    style={'textAlign':'center'}
    ), # page title
    
    html.Div(children='''
        Historic information and forecasting
    ''',
    style={'textAlign':'center'}
    ), # page subtitle
       
    html.Div([
        html.Label('Buildings:'),
            dcc.Checklist(
            id = 'Building',
            options=[
                {'label': 'Total', 'value': 'Total'},
                {'label': 'Central', 'value': 'Central'},
                {'label': 'Civil', 'value': 'Civil'},
                {'label': 'South T', 'value': 'South'},
                {'label': 'North T', 'value': 'North'}
                ],
                values=['South']
                ), # checklist and labels for buildings
            ],style={'width': '120%', 'margin-left':100}), # change the style to fit the webpage
    
    html.Div([
        html.Label('Forecast model:'),
        dcc.RadioItems(
            id = 'F_Model',
            options=[
                {'label': 'LR', 'value': 'LR'},
                {'label': 'NN', 'value': 'NN'},
                {'label': 'RF', 'value': 'RF'}
                ],
            value='LR',
            labelStyle={'display': 'block'} # change the display style to vertical
            ), # choice of forecasting model
    
        html.Label('Forecast scale:'),
        dcc.RadioItems(
            id = 'F_Scale',
            options=[
                {'label': 'Hourly', 'value': 'H'},
                {'label': 'Daily', 'value': 'D'},
                {'label': 'Weekly', 'value': 'W'},
                {'label': 'Monthly', 'value': 'M'}    
                ],
            value='H',
            labelStyle={'display': 'block'} # change the display style to vertical
            ) # choice of forecasting scale
        ],style={'width': '30%','margin-top':150,'float': 'right', 'display':'inline-block'}), # style parameters
    
    dcc.Graph(id='Historic_forecast',style={'width':'70%','float':'left','display':'inline-block'}), #id,style parameters for the graph
    
    ])
    
@app.callback(
        Output('Historic_forecast','figure'),
        [Input('F_Scale','value'),
         Input('F_Model','value'),
         Input('Building','values')
         ]) # define inputs and output
    

# define the function for updating the graph     
def update_graph(scale,model,buildings):
    
    # import modules for forecasting
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    import datetime
    
    traces = []
    for building in buildings:
        new_df = df[['Power_kW_'+building]]
        new_df.rename(columns={'Power_kW_'+building:'Power_kW'}, inplace=True)
        # if the timescale is hour
        if scale == 'H':
            hourly_df = new_df.copy() # copy of the df
            # create the features: hour, day, weekday, month, year, power-1
            hourly_df['hour']=hourly_df.index.hour
            hourly_df['day']=hourly_df.index.day
            hourly_df['weekday']=hourly_df.index.weekday
            hourly_df['month']=hourly_df.index.month
            hourly_df['year']=hourly_df.index.year
            hourly_df['power-1']=hourly_df.Power_kW.shift(1)
            hourly_df=hourly_df.dropna() # drop NAN values
            
            # choose the values for training and testing
            Y=hourly_df.values[:,0]
            X=hourly_df.values[:,[1,2,3,4,5,6]]
            X_train,X_test,y_train,y_test = train_test_split(X,Y)
            
            # shift the last existing time to next one
            Next_index = hourly_df.index[-1]+datetime.timedelta(hours=1)
            
            # generate features from the next timescale
            Next_scale = [[Next_index.hour,
                           Next_index.day,
                           Next_index.dayofweek,
                           Next_index.month,
                           Next_index.year,
                           hourly_df['power-1'][-1]]]
        
            # forecasting by LR model
            if model == 'LR':
                regr = LinearRegression()
                regr.fit(X_train,y_train)
                Next_power = regr.predict(Next_scale)
                
            # forecasting by NN model
            if model == 'NN':
                mlp = MLPRegressor(hidden_layer_sizes = (40))
                mlp.fit(X_train,y_train)
                Next_power = mlp.predict(Next_scale)
            
            # forecasting by RF model
            if model == 'RF':
                parameters = {'bootstrap': True,
                              'min_samples_leaf': 8,
                              'n_estimators': 600, # number of trees
                              'min_samples_split': 12,
                              'max_features': 'sqrt',
                              'max_depth': 10,
                              'max_leaf_nodes': None}
                RF_model = RandomForestRegressor(**parameters)
                RF_model.fit(X_train, y_train)
                Next_power = RF_model.predict(Next_scale)
                
            # choose last 24 hours for graph
            hourly_df = new_df.iloc[len(df)-24:,:]
            
            # generate x and y for plotting
            plot_x = hourly_df.index
            plot_y = hourly_df['Power_kW'].values
            
            traces.append(go.Scatter(
                        x = [plot_x[-1],Next_index], 
                        y = [plot_y[-1],Next_power[0]],
                        name = building+'_forecast',
                        line = dict(dash='dash')
                        ))
            traces.append(go.Scatter(
                        x = plot_x, 
                        y = plot_y,
                        name = building+'_historical',
                        mode = 'lines+markers'
                        ))
    
        # if the timescale is day    
        if scale == 'D':
            daily_df = new_df.resample('D').sum() # resample the df to daily_df
#        for building in buildings:
#            daily_df = daily_df[['Power_kW_'+building]]
#            daily_df.rename(columns={'Power_kW_'+building:'Power_kW'}, inplace=True)
        
            # create the features: day, weekday, month, year, power-1
            daily_df['day']=daily_df.index.day
            daily_df['weekday']=daily_df.index.weekday
            daily_df['month']=daily_df.index.month
            daily_df['year']=daily_df.index.year
            daily_df['power-1']=daily_df.Power_kW.shift(1)
            daily_df=daily_df.dropna() # drop NAN values
        
            # choose the values for training and testing
            Y=daily_df.values[:,0]
            X=daily_df.values[:,[1,2,3,4,5]]
            X_train,X_test,y_train,y_test = train_test_split(X,Y)
        
            # shift the last existing time to next one
            Next_index = daily_df.index[-1]+datetime.timedelta(days=1)
            
            # generate features from the next timescale
            Next_scale = [[Next_index.day,
                           Next_index.dayofweek,
                           Next_index.month,
                           Next_index.year,
                           daily_df['power-1'][-1]]]
            
            # forecasting by LR model
            if model == 'LR':
                regr = LinearRegression()
                regr.fit(X_train,y_train)
                Next_power = regr.predict(Next_scale)
                
            # forecasting by NN model    
            if model == 'NN':
                mlp = MLPRegressor(hidden_layer_sizes = (40))
                mlp.fit(X_train,y_train)
                Next_power = mlp.predict(Next_scale)
                
            # forecasting by RF model
            if model == 'RF':
                parameters = {'bootstrap': True,
                              'min_samples_leaf': 8,
                              'n_estimators': 600, # number of trees
                              'min_samples_split': 12,
                              'max_features': 'sqrt',
                              'max_depth': 10,
                              'max_leaf_nodes': None}
                RF_model = RandomForestRegressor(**parameters)
                RF_model.fit(X_train, y_train)
                Next_power = RF_model.predict(Next_scale)
                
            # choose last 14 days for the graph
            daily_df = daily_df.iloc[len(daily_df)-14:,:]
            
            plot_x = daily_df.index
            plot_y = daily_df['Power_kW'].values
            
            traces.append(go.Scatter(
                        x = [plot_x[-1],Next_index], 
                        y = [plot_y[-1],Next_power[0]],
                        name = building+'_forecast',
                        line = dict(dash='dash')
                        ))
            traces.append(go.Scatter(
                        x = plot_x, 
                        y = plot_y,
                        name = building+'_historical',
                        mode = 'lines+markers'
                        ))

        # if the timescale is weekly
        if scale == 'W':
            weekly_df = new_df.resample('W').sum() # resample the df to weekly_df
#        for building in buildings:
#            weekly_df = weekly_df[['Power_kW_'+building]]
#            weekly_df.rename(columns={'Power_kW_'+building:'Power_kW'}, inplace=True) 
#        
            # create features: day, week, month, year, power-1
            weekly_df['day']=weekly_df.index.day
            weekly_df['week']=weekly_df.index.week
            weekly_df['month']=weekly_df.index.month
            weekly_df['year']=weekly_df.index.year
            weekly_df['power-1']=weekly_df.Power_kW.shift(1)
            weekly_df=weekly_df.dropna() # drop NAN values
            
            # get the values for training 
            Y=weekly_df.values[:,0]
            X=weekly_df.values[:,[1,2,3,4,5]]
            X_train,X_test,y_train,y_test = train_test_split(X,Y)
            
            # shift the last existing time to next one
            Next_index = weekly_df.index[-1]+datetime.timedelta(days=7)
            
            # generate features from the next timescale for forecasting
            Next_scale = [[Next_index.day,
                           Next_index.isocalendar()[1],
                           Next_index.month,
                           Next_index.year,
                           weekly_df['power-1'][-1]]]
            
            # forecasting by LR model
            if model == 'LR':
                regr = LinearRegression()
                regr.fit(X_train,y_train)
                Next_power = regr.predict(Next_scale)
                
            # forecasting by NN model
            if model == 'NN':
                mlp = MLPRegressor(hidden_layer_sizes = (40))
                mlp.fit(X_train,y_train)
                Next_power = mlp.predict(Next_scale)
                
            # forecasting by RF model
            if model == 'RF':
                parameters = {'bootstrap': True,
                              'min_samples_leaf': 8,
                              'n_estimators': 600, # number of trees
                              'min_samples_split': 12,
                              'max_features': 'sqrt',
                              'max_depth': 10,
                              'max_leaf_nodes': None}
                RF_model = RandomForestRegressor(**parameters)
                RF_model.fit(X_train, y_train)
                Next_power = RF_model.predict(Next_scale)
        
            # choose last 8 weeks for the graph
            weekly_df = weekly_df.iloc[len(weekly_df)-8:,:]
            plot_x = weekly_df.index
            plot_y = weekly_df['Power_kW'].values
            
            traces.append(go.Scatter(
                        x = [plot_x[-1],Next_index], 
                        y = [plot_y[-1],Next_power[0]],
                        name = building+'_forecast',
                        line = dict(dash='dash')
                        ))
            traces.append(go.Scatter(
                        x = plot_x, 
                        y = plot_y,
                        name = building+'_historical',
                        mode = 'lines+markers'
                        ))

        # if the timescale is monthly
        if scale == 'M':
            monthly_df = new_df.resample('M').sum() # resample the df to monthly_df
#        for building in buildings:
#            weekly_df = weekly_df[['Power_kW_'+building]]
#            weekly_df.rename(columns={'Power_kW_'+building:'Power_kW'}, inplace=True) 
        
            # create features: day, month, year, power-1
            monthly_df['day']=monthly_df.index.day
            monthly_df['month']=monthly_df.index.month
            monthly_df['year']=monthly_df.index.year
            monthly_df['power-1']=monthly_df.Power_kW.shift(1)
            monthly_df=monthly_df.dropna() # drop NAN 
            
            # data for training 
            Y=monthly_df.values[:,0]
            X=monthly_df.values[:,[1,2,3,4]]
            X_train,X_test,y_train,y_test = train_test_split(X,Y)
            
            # shift the last existing time to next one
            from dateutil.relativedelta import relativedelta
            Next_index = monthly_df.index[-1]+relativedelta(months=1)
            
            # generate features from the next timescale for forecasting
            Next_scale = [[Next_index.day,
                           Next_index.month,
                           Next_index.year,
                           monthly_df['power-1'][-1]]]
            
            # forecasting by LR model
            if model == 'LR':
                regr = LinearRegression()
                regr.fit(X_train,y_train)
                Next_power = regr.predict(Next_scale)
                
            # forecasting by NN model
            if model == 'NN':
                mlp = MLPRegressor(hidden_layer_sizes = (40))
                mlp.fit(X_train,y_train)
                Next_power = mlp.predict(Next_scale)
            
            # forecasting by RF model
            if model == 'RF':
                parameters = {'bootstrap': True,
                              'min_samples_leaf': 8,
                              'n_estimators': 600, # number of trees
                              'min_samples_split': 12,
                              'max_features': 'sqrt',
                              'max_depth': 10,
                              'max_leaf_nodes': None}
                RF_model = RandomForestRegressor(**parameters)
                RF_model.fit(X_train, y_train)
                Next_power = RF_model.predict(Next_scale)
        
            # choose last 6 months for the graph
            monthly_df = monthly_df.iloc[len(monthly_df)-6:,:]
            plot_x = monthly_df.index
            plot_y = monthly_df['Power_kW'].values
            
            traces.append(go.Scatter(
                        x = [plot_x[-1],Next_index], 
                        y = [plot_y[-1],Next_power[0]],
                        name = building+'_forecast',
                        line = dict(dash='dash')
                        ))
            traces.append(go.Scatter(
                        x = plot_x, 
                        y = plot_y,
                        name = building+'_historical',
                        mode = 'lines+markers'
                        ))
    
    # plot the graph    
    return {
            'data':traces,
            'layout':{
                    'xaxis':{'title':'Time'},
                    'yaxis':{'title':'Power[kW]'},
                    }
            }
        
    
if __name__ == '__main__':
    app.run_server(debug=True)