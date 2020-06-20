
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.cluster import KMeans
#mergedf = (df1.merge(df2, left_on='Conv_ID', right_on='Conv_ID').reindex(columns=['Conv_ID', 'Conv_Date', 'User_ID', 'Channel', 'IHC_Conv','Revenue']))
#print(len(mergedf.index))
#mergedf.to_csv(r'rge_C.csv')
#print(mergedf.describe())
#mergedf.to_csv(r'c:\Users\anves\Downloads\MergeC.csv.txtv')

#getting current folder directory
here = os.path.dirname(os.path.abspath(__file__))

#get the combine dataframe with respect to Conv_ID
def mergedf():
    df1 = pd.read_csv(os.path.join(here, 'table_A_conversions.csv.txt'))
    df2 = pd.read_csv(os.path.join(here, 'table_B_attribution.csv.txt'))
    mergedf = (df1.merge(df2, left_on='Conv_ID', right_on='Conv_ID').reindex(columns=['Conv_ID', 'Conv_Date', 'User_ID', 'Channel', 'IHC_Conv','Revenue']))
    mergedf.Revenue = mergedf.Revenue.round().astype(int)
    mergedf.to_csv(os.path.join(here, 'Merge_c.csv.txt'))

#Correlation for the given dataframe
def corrrelplot(df):
    corr = df.corr()
    sns.pairplot(df)
    plt.show()

#Plot for the merged dataframe table
def Merge_c_plots():
    df = pd.read_csv(os.path.join(here, 'Merge_c.csv.txt'))
    corrrelplot(df)

#Transfrom merged dataframe with respect to channels
def matrixTransform(mergedf):
    data = {'Conv_ID': [], 'Conv_Date': [], 'User_ID': [], 'IHC_Conv': [] ,'Revenue': []}
    channels = mergedf.Channel.unique()
    for i in channels:
        data[i] = []
    for index, row in mergedf.iterrows():
            data['Conv_ID'] = np.append(data['Conv_ID'],row['Conv_ID'])
            data['Conv_Date'] = np.append(data['Conv_Date'],row['Conv_Date'])
            data['IHC_Conv'] = np.append(data['IHC_Conv'],row['IHC_Conv'])
            data['Revenue'] = np.append(data['Revenue'],row['Revenue'])
            data['User_ID'] = np.append(data['User_ID'],row['User_ID'])
            print(index)
            for i in channels:
                if i == row['Channel']:
                    data[i] = np.append(data[i],1)
                else:
                    data[i] = np.append(data[i],0)
    transdf = pd.DataFrame(data)
    transdf.to_csv(os.path.join(here, 'transD.csv.txt'))

#Creates Cohortperiod based on users first purchase
def cohort_period(df):
    """
    Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.
    
    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['User_ID', 'Conv_Date', inplace=True)
        df = df.groupby('User_ID').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df    


def table_A_cohorts_user_retention():
    #loads table A data into data frame
    df = pd.read_csv(os.path.join(here, 'table_A_conversions.csv.txt'))
    #Convert revenue values to int
    df.Revenue = df.Revenue.round().astype(int)
    #remove rows with missing data
    df = df.dropna()
    #add conversion period from date for later grouping
    df['Conv_Period'] = df.Conv_Date.apply(lambda x:  datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
    df.set_index('User_ID', inplace=True)
    #determine user's cohort group based on first conversion date
    df['CohortGroup'] = df.groupby(level=0)['Conv_Date'].min().apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
    df.reset_index(inplace=True)
    grouped = df.groupby(['CohortGroup', 'Conv_Period'])
    #aggregating users,conversions,revenue on monthly basis
    cohorts = grouped.agg({'User_ID': pd.Series.nunique,
                       'Conv_ID': pd.Series.nunique,
                       'Revenue': np.sum})
    cohorts.rename(columns={'User_ID': 'TotalUsers',
                        'Conv_ID': 'TotalOrders'}, inplace=True)
    cohorts = cohorts.groupby(level=0).apply(cohort_period)
    # reindex the DataFrame
    cohorts.reset_index(inplace=True)
    cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)
    # create a Series holding the total size of each CohortGroup
    cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
    cohorts['TotalUsers'].head()
    cohorts['TotalUsers'].unstack(0).head()
    user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
    user_retention.head(10)
    user_retention[['2017-06', '2017-07', '2017-11','2017-12','2018-01']].plot(figsize=(10,5))
    plt.title('Cohorts: User Retention')
    plt.xticks(np.arange(1, 12.1, 1))
    plt.xlim(1, 12)
    plt.ylabel('% of Cohort Purchasing')
    #       Seaborn for heatmap
    sns.set(style='white')
    plt.figure(figsize=(12, 8))
    plt.title('Cohorts: User Retention')
    sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%')
    plt.show()


def table_A_time_series_analyis():
    #loads table A data into data frame,Time-based indexing on Conv_Date
    df = pd.read_csv(os.path.join(here, 'table_A_conversions.csv.txt'),index_col=0, parse_dates=True)
    #create columns for year, month, weekday name
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['Weekday Name'] = df.index.day_name()
    # seaborn style default figure size
    sns.set(rc={'figure.figsize':(11, 4)})
    df['Revenue'].plot(linewidth=0.5,label ='Revenue over time')
    plt.show()
    df['Revenue'].resample('M').sum().plot(linewidth=1, label='Total revenue summed monthly')
    plt.show()
    data_columns = ['User_ID','Conv_ID']
    df[data_columns].resample('M').count().plot(linewidth=1, label='Number of Users, Conversions Monthly')
    plt.show()

    
#Creates CSV file with channels related summary
def Channels_Impact():
    df1 = pd.read_csv(os.path.join(here, 'table_A_conversions.csv.txt'))
    df2 = pd.read_csv(os.path.join(here, 'table_B_attribution.csv.txt'))
    df = (df1.merge(df2, left_on='Conv_ID', right_on='Conv_ID').reindex(columns=['Conv_ID', 'Conv_Date', 'User_ID', 'Channel', 'IHC_Conv','Revenue']))
    channels = df.Channel.unique()
    Channels_impact_data = {'channels':[],'users_count':[],'conversions_count':[],'revenue_max':[],'revenue_firstQuartile':[],'revenue_thirdQuartile':[],'revenue_median':[],'total_revenue':[],'IHC_firstQuartile':[],'IHC_secondQuartile':[],'IHC_thirdQuartile':[],'IHC_max':[],'IHC_min':[],'IHC_median':[]}
    for i in channels:
       newdf =  df.loc[df['Channel'] == i]
       Channels_impact_data['channels'] = np.append(Channels_impact_data['channels'], i)
       Channels_impact_data['users_count'] = np.append(Channels_impact_data['users_count'], newdf.User_ID.count() if newdf.User_ID.count() else 0)
       Channels_impact_data['conversions_count'] = np.append(Channels_impact_data['conversions_count'], newdf.Conv_ID.count() if newdf.Conv_ID.count() else 0)
       Channels_impact_data['revenue_max'] = np.append(Channels_impact_data['revenue_max'],newdf.Revenue.max() if newdf.Revenue.max() else 0)
       Channels_impact_data['revenue_firstQuartile'] = np.append(Channels_impact_data['revenue_firstQuartile'],newdf.Revenue.quantile(.25) if newdf.Revenue.quantile(.25) else 0)
       Channels_impact_data['revenue_thirdQuartile'] = np.append(Channels_impact_data['revenue_thirdQuartile'],newdf.Revenue.quantile(.75) if newdf.Revenue.quantile(.75) else 0)
       Channels_impact_data['revenue_median'] = np.append(Channels_impact_data['revenue_median'],newdf.Revenue.median() if newdf.Revenue.median() else 0)
       Channels_impact_data['total_revenue'] = np.append(Channels_impact_data['total_revenue'], newdf.Revenue.sum() if newdf.Revenue.sum() else 0)
       Channels_impact_data['IHC_firstQuartile'] = np.append(Channels_impact_data['IHC_firstQuartile'],newdf.IHC_Conv.quantile(.25) if newdf.IHC_Conv.quantile(.25) else 0)
       Channels_impact_data['IHC_secondQuartile'] = np.append(Channels_impact_data['IHC_secondQuartile'],newdf.IHC_Conv.quantile(.50) if newdf.IHC_Conv.quantile(.50) else 0)
       Channels_impact_data['IHC_thirdQuartile'] = np.append(Channels_impact_data['IHC_thirdQuartile'],newdf.IHC_Conv.quantile(.75) if newdf.IHC_Conv.quantile(.75) else 0)
       Channels_impact_data['IHC_max'] = np.append(Channels_impact_data['IHC_max'],newdf.IHC_Conv.max() if newdf.IHC_Conv.max() else 0)
       Channels_impact_data['IHC_min'] = np.append(Channels_impact_data['IHC_min'],newdf.IHC_Conv.min() if newdf.IHC_Conv.min() else 0)
       Channels_impact_data['IHC_median'] = np.append(Channels_impact_data['IHC_median'],newdf.IHC_Conv.median() if newdf.IHC_Conv.median() else 0)
    impact_df = pd.DataFrame(Channels_impact_data)
    impact_df.to_csv(os.path.join(here, 'channels_impact.csv.txt'))



#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


#Revenue vs Frequency
def revenueVsFreq(df_user):
    df_graph = df_user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=df_graph.query("Segment == 'Low-Value'")['Frequency'],
            y=df_graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'Mid-Value'")['Frequency'],
            y=df_graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'High-Value'")['Frequency'],
            y=df_graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Frequency"},
            title='Segments'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.plot(fig)


#Revenue Recency plot
def revenueVsrecency(df_user):
    df_graph = df_user.query("Revenue < 50000 and Frequency < 2000")
    plot_data = [
        go.Scatter(
            x=df_graph.query("Segment == 'Low-Value'")['Recency'],
            y=df_graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'Mid-Value'")['Recency'],
            y=df_graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'High-Value'")['Recency'],
            y=df_graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.plot(fig)

#Recency Frequency plot
def recencyVsfrequency(df_user):
    df_graph = df_user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=df_graph.query("Segment == 'Low-Value'")['Recency'],
            y=df_graph.query("Segment == 'Low-Value'")['Frequency'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'Mid-Value'")['Recency'],
            y=df_graph.query("Segment == 'Mid-Value'")['Frequency'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'High-Value'")['Recency'],
            y=df_graph.query("Segment == 'High-Value'")['Frequency'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Frequency"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.plot(fig)

#Generates RFM Clustering (Recency,Frequency,Revenue(Monetary val)) plots with low,mid,high segments
#Low - not frequent customers, Mid- moderately frequent and generates some revenue, High - high value customers
def customer_segmentation():
    df = pd.read_csv(os.path.join(here, 'table_A_conversions.csv.txt'))
    df['Conv_Date'] = pd.to_datetime(df['Conv_Date'])
    df_user = pd.DataFrame(df['User_ID'].unique())
    df_user.columns = ['User_ID']
    df_max_conversion = df.groupby('User_ID').Conv_Date.max().reset_index()
    df_max_conversion.columns = ['User_ID','MaxConversionDate']
    df_max_conversion['Recency'] = (df_max_conversion['MaxConversionDate'].max() - df_max_conversion['MaxConversionDate']).dt.days
    df_user = pd.merge(df_user, df_max_conversion[['User_ID','Recency']], on='User_ID')
    #print(df_user.head())
    #print(df_user.Recency.describe())
    # Avg of 174 days from describe
    plot_data = [go.Histogram(
        x=df_user['Recency']
    )]
    plot_layout = go.Layout(
        title='Recency'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    #pyoff.plot(fig)
    # k-means clustering
    sse={}
    df_recency = df_user[['Recency']]
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
        df_recency["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ 
    #plt.figure()
    #plt.plot(list(sse.keys()), list(sse.values()))
    #plt.xlabel("Number of cluster")
    #plt.show()
    #4 clusters are optimal
    #build 4 clusters for recency and add it to dataframe
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df_user[['Recency']])
    df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])
    df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)
    #print(df_user.describe())
    print(df_user.groupby('RecencyCluster')['Recency'].describe())
    return

    # frequency
    #get order counts for each user and create a dataframe with it
    df_frequency = df.groupby('User_ID').Conv_Date.count().reset_index()
    df_frequency.columns = ['User_ID','Frequency']
    #add this data to our main dataframe
    df_user = pd.merge(df_user, df_frequency, on='User_ID')
    #plot the histogram
    plot_data = [
    go.Histogram(
        x= df_user.query('Frequency < 1000')['Frequency']
    )
    ]

    plot_layout = go.Layout(
        title='Frequency'
    )
    #frequency_plot
    #fig = go.Figure(data=plot_data, layout=plot_layout)
    #pyoff.plot(fig)
    #k-means
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df_user[['Frequency']])
    df_user['FrequencyCluster'] = kmeans.predict(df_user[['Frequency']])

    #order the frequency cluster
    df_user = order_cluster('FrequencyCluster', 'Frequency',df_user,True)

    #see details of each cluster
    df_user.groupby('FrequencyCluster')['Frequency'].describe()

    #Revenue
    #calculate revenue for each customer
    
    df_revenue = df.groupby('User_ID').Revenue.sum().reset_index()

    #merge it with our main dataframe
    df_user = pd.merge(df_user, df_revenue, on='User_ID')

    #plot the histogram
    plot_data = [
    go.Histogram(
        x=df_user.query('Revenue < 10000')['Revenue']
    )
    ]

    plot_layout = go.Layout(
        title='Monetary Value'
    )
    #fig = go.Figure(data=plot_data, layout=plot_layout)
    #pyoff.plot(fig)
    
    #apply clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df_user[['Revenue']])
    df_user['RevenueCluster'] = kmeans.predict(df_user[['Revenue']])


    #order the cluster numbers
    df_user = order_cluster('RevenueCluster', 'Revenue',df_user,True)

    #show details of the dataframe
    df_user.groupby('RevenueCluster')['Revenue'].describe()

    #calculate overall score and use mean() to see details
    df_user['OverallScore'] = df_user['RecencyCluster'] + df_user['FrequencyCluster'] + df_user['RevenueCluster']
    df_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()

    # segmenting 0 to 2: Low Value 3 to 4: Mid Value 5+: High Value
    df_user['Segment'] = 'Low-Value'
    df_user.loc[df_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    df_user.loc[df_user['OverallScore']>4,'Segment'] = 'High-Value' 

    #revenueVsFreq(df_user)
    #revenueVsrecency(df_user)
    recencyVsfrequency(df_user)








