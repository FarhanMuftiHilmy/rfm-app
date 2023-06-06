import pickle 
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly_express as px
import plotly.graph_objects as go

import datetime as dt
from sklearn.cluster import KMeans
import calendar
import plotly.io as pio


import streamlit_authenticator
import mysql.connector as mc

import streamlit as st

import numpy as np

st.set_page_config(page_title="RFM Dashboard Analysis", page_icon="📈", layout="wide")

@st.cache_data
def getUser():
    mysqldb_conn= mc.connect(host="localhost",user="root",password="hilmy148",database="db_rfm")
    get_users_query = "SELECT * FROM users"
    cursor = mysqldb_conn.cursor()
    cursor.execute(get_users_query)
    rows = cursor.fetchall()  
    cursor.close()
    mysqldb_conn.close()
    return rows
# credentials = {"usernames":{}}
# users = getUser()
# for username, password, name in users:
#     user_dict = {"name":name,"password":password}
#     credentials["usernames"].update({username:user_dict})


# # ----USER-AUTH

# authenticator = streamlit_authenticator.Authenticate(credentials, "rfm_dashboard", "abcdef", cookie_expiry_days=30)

# ----USER-AUTH
names = ["Admin 1", "Admin 2"]
usernames = ["admin1", "admin2"]
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)
    

# users =  database.

credentials = {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                }            
            }
        }

authenticator = streamlit_authenticator.Authenticate(credentials, "rfm_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")




if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")


if authentication_status:
        
#-----------------------------LOGOUT-----------------------------



        
        #JSON WEB TOKEN COOKIES to reauthenticate

        # caching dataframe -> read data from short term memory instead of excel
    # @st.cache_data
    # def get_data_from_excel():
    #     df = pd.read_excel(
    #         io='sales_data.xlsx',
    #         engine='openpyxl',
    #         sheet_name='Sales',
    #         skiprows=3,
    #         usecols='B:S',
    #         nrows=1000,
    #     )
    #     # add new column "hour" to dataframe
    #     df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    #     df["recency"] = pd.to_datetime(df["Date"])


    #     # df['tahun'] = pd.DatetimeIndex(df['tanggal']).year
    #     # df['bulan'] = pd.DatetimeIndex(df['tanggal']).month
    #     return df
    # df = get_data_from_excel()


    #OPTION2 : USING CSV DATA -> WHY NOT MODIFY THE EXCEL ONE
    @st.cache_data
    def get_data_from_csv():
        # conn = sqlite3.connect('customers.db') 
        # sql_query = pd.read_sql_query("SELECT * FROM data", conn)
        # mysqldb_conn= mc.connect(host="localhost",user="root",password="hilmy148",database="db_rfm")
        # sql_query = pd.read_sql_query("SELECT * FROM customer_data", mysqldb_conn)
        # dfsql = pd.DataFrame(sql_query)
        dfsql = pd.read_csv("data.csv", encoding='unicode_escape')
        dfsql["InvoiceDate"] = pd.to_datetime(dfsql["InvoiceDate"])
        dfsql['tahun'] = pd.DatetimeIndex(dfsql['InvoiceDate']).year
        dfsql['bulan'] = pd.DatetimeIndex(dfsql['InvoiceDate']).month
        dfsql['tanggal'] = pd.DatetimeIndex(dfsql['InvoiceDate']).date
        dfsql["BulanTahun"] =  pd.to_datetime(dfsql['InvoiceDate']).dt.strftime('%Y-%m')
        dfsql["sales"] = dfsql["UnitPrice"]*dfsql["Quantity"]
        dfsql["JumlahTransaksi"] = dfsql["InvoiceDate"]
        dfsql["JumlahPelanggan"] = dfsql["CustomerID"]
        dfsql["Recency"] = dfsql["InvoiceDate"]
        dfsql["Frequency"] = dfsql["CustomerID"]
        dfsql["Monetary"] = dfsql["sales"]
        return dfsql
    dfsql = get_data_from_csv()
    # print("Dataframe from sql: ", dfsql["sales"])




    #OPTION2 : USING SQL DATA 
    # @st.cache_data
    # def get_data_from_sql():
    #     # conn = sqlite3.connect('customers.db') 
    #     # sql_query = pd.read_sql_query("SELECT * FROM data", conn)
    #     mysqldb_conn= mc.connect(host="localhost",user="root",password="hilmy148",database="db_rfm")
    #     sql_query = pd.read_sql_query("SELECT * FROM customer_data", mysqldb_conn)
    #     dfsql = pd.DataFrame(sql_query)
    #     dfsql["InvoiceDate"] = pd.to_datetime(dfsql["InvoiceDate"])
    #     dfsql['tahun'] = pd.DatetimeIndex(dfsql['InvoiceDate']).year
    #     dfsql['bulan'] = pd.DatetimeIndex(dfsql['InvoiceDate']).month
    #     dfsql['tanggal'] = pd.DatetimeIndex(dfsql['InvoiceDate']).date
    #     dfsql["BulanTahun"] =  pd.to_datetime(dfsql['InvoiceDate']).dt.strftime('%Y-%m')
    #     dfsql["sales"] = dfsql["UnitPrice"]*dfsql["Quantity"]
    #     dfsql["JumlahTransaksi"] = dfsql["InvoiceDate"]
    #     dfsql["JumlahPelanggan"] = dfsql["CustomerID"]
    #     dfsql["Recency"] = dfsql["InvoiceDate"]
    #     dfsql["Frequency"] = dfsql["CustomerID"]
    #     dfsql["Monetary"] = dfsql["sales"]
    #     return dfsql
    # dfsql = get_data_from_sql()
    # print("Dataframe from sql: ", dfsql["sales"])


    # st.title(":bar_chart: RFM Dashboard")
    st.title("RFM Dashboard")
    st.markdown("##")

    
    #-----------------------------FILTER-----------------------------
    def logout(name, authenticator: streamlit_authenticator.Authenticate):
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Welcome {name}")
    logout(name, authenticator)
    st.sidebar.header("Please Filter here:")
    def filterDashboard(dfsql: pd.DataFrame):
        tahun = st.sidebar.multiselect(
            "Select the year: ",
            options=dfsql["tahun"].unique(),
            default=dfsql["tahun"].unique()
        )
        dfsql_selection_by_year = dfsql.query(
            "tahun == @tahun"
        )
        bulan = st.sidebar.multiselect(
            "Select the month: ",
            options=dfsql_selection_by_year["bulan"].sort_values().unique(),
            default=dfsql_selection_by_year["bulan"].sort_values().unique()
        )
        dfsql_selection_by_month = dfsql_selection_by_year.query(
            "bulan == @bulan"
        )
        country = st.sidebar.multiselect(
            "Select the City:",
            options=dfsql_selection_by_month["Country"].unique(),
            default=dfsql_selection_by_month["Country"].unique()
        )        

        dfsql_selection = dfsql_selection_by_month.query(
            "Country == @country"
        )
        return bulan,dfsql_selection
    bulan, dfsql_selection = filterDashboard(dfsql)
    # print("Raw Dataframe: ", dfsql_selection)
    
 
   
    # print("""MINIMUM DATE
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -
    # -""", "DONE")

    # st.dataframe(dfsql_selection)

    #-----------------------------KPI [Key Performance Indicator]-----------------------------

    if dfsql_selection.empty is False:

        
          # count_row_with_nan = dfsql_rfm.isnull().any(axis=1).sum()
            # print ('Count rows with NaN: ' + str(count_row_with_nan))
            # count_neg_sales = (dfsql_selection['UnitPrice'] < 0).sum()
            # count_neg_quantity = (dfsql_selection['Quantity'] < 0).sum()

            # print ('Count rows with negative values of UnitPrice: ' + str(count_neg_sales))
            # print ('Count rows with negative values of Quantity: ' + str(count_neg_quantity))
        def runDataPreprocessingStep():
            neg_quantity_and_price = dfsql_selection[(dfsql_selection['Quantity'] < 0) |
                                            (dfsql_selection['UnitPrice'] < 0)].index
            dfsql_selection.drop(neg_quantity_and_price, inplace=True)
            PRESENT = dt.datetime(2011, 12, 31)
            dfsql_rfm = dfsql_selection.groupby('CustomerID').agg({'Recency': lambda date: (PRESENT - date.max()).days, 
                                                        'Frequency': lambda num: len(num),
                                                        'Monetary': lambda price: price.sum()}).reset_index()            
            cols = list(dfsql_rfm.columns)
            cols.remove("CustomerID")
            for col in cols:
                col_zscore = col + '_zscore'
                dfsql_rfm[col_zscore] = (dfsql_rfm[col] - dfsql_rfm[col].mean())/dfsql_rfm[col].std(ddof=0)
            dfsql_rfm["Outlier"] = ((abs(dfsql_rfm["Monetary_zscore"])>3).astype(int) |
                            (abs(dfsql_rfm["Frequency_zscore"])>3) | 
                             (abs(dfsql_rfm["Recency_zscore"])>3)).astype(int)
            outlier = dfsql_rfm[dfsql_rfm["Outlier"] == 1].index
            dfsql_rfm.drop(outlier, inplace=True)
            print("After zscore: ", dfsql_rfm['CustomerID'].count())
            return dfsql_rfm
        dfsql_rfm = runDataPreprocessingStep()

        def showKPI():
            total_sales = int((dfsql_selection["UnitPrice"]*dfsql_selection["Quantity"]).sum())
            average_sale_by_transaction = round((dfsql_selection["UnitPrice"]*dfsql_selection["Quantity"]).mean(), 2)
            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Sales total:")
                st.subheader(f"US $ {total_sales:,}")
            with right_column:
                st.subheader("Average sales by transaction:")
                st.subheader(f"US $ {average_sale_by_transaction}")
            st.markdown("---")
            print("Sales count :", total_sales)
        showKPI()
          
        def showLineChart():
            if len(dfsql_selection["bulan"].unique()) == 1 and len(dfsql_selection["tahun"].unique()) == 1:
                sales_by_date = dfsql_selection.groupby(['tanggal']).agg({'sales': 'sum'}).reset_index()
                fig_product_sales_by_date = px.line(
                    sales_by_date,
                    x="tanggal",
                    y="sales",
                    title = f"<b> Total Penjualan per Bulan {calendar.month_name[bulan[0]]}</b>",
                    color_discrete_sequence=["#7f5539"]
                )
                fig_product_sales_by_date.add_trace(go.Scatter(
                    x=[sales_by_date['tanggal'][0], sales_by_date.iloc[-1]['tanggal']], y=[sales_by_date['sales'][0], sales_by_date.iloc[-1]['sales']],
                    line_color='#b08968',
                    name='Naik/Turun',
                    mode='lines',
                    line=dict(dash='dash')
                ))
                st.plotly_chart(fig_product_sales_by_date)
            else:
                sales_by_month_year = dfsql_selection.groupby(['BulanTahun']).agg({'sales': 'sum'}).reset_index()
                fig_product_sales_by_month_year = px.line(
                    sales_by_month_year,
                    x="BulanTahun",
                    y="sales",
                    title = f"<b> Total Penjualan Per Tahun dan Bulan</b>",
                    color_discrete_sequence=['#b08968']
                )
                first_x = sales_by_month_year.loc[0,"BulanTahun"]
                last_x = sales_by_month_year.iloc[-1]["BulanTahun"]
                fig_product_sales_by_month_year.update_layout(
                    xaxis_title= f"Periode {first_x} ~ {last_x}",
                    yaxis_title="Total",
                )
                fig_product_sales_by_month_year.add_trace(go.Scatter(
                    x=[sales_by_month_year['BulanTahun'][0], sales_by_month_year.iloc[-1]['BulanTahun']], y=[sales_by_month_year['sales'][0], sales_by_month_year.iloc[-1]['sales']],
                    line_color='#b08968',
                    name='Naik/Turun',
                    mode='lines',
                    line=dict(dash='dash')
                ))
                st.plotly_chart(fig_product_sales_by_month_year, use_container_width=True)
        showLineChart()

        # @st.cache_data
        # def convert_df_raw_data(df):
        #     return df.to_csv().encode('utf-8')
        # csv = convert_df_raw_data(dfsql_selection)
        # st.download_button('Download Dataframe of Raw Data as CSV', csv, 'raw_data_dataframe.csv', 'text/csv')

        def runModelDevelopment(dfsql_rfm):
            data = dfsql_rfm[['Recency','Frequency','Monetary']]
            k_means = KMeans(n_clusters=3, random_state=42)
            k_means.fit(data)
            labels = k_means.labels_
            dfsql_rfm['ClusterLabels'] = labels
            
            x_val = 'Recency'
            y_val = 'Frequency'
            z_val = 'Monetary'

            fig_rfm_kmeans_clustering_3d = px.scatter_3d(dfsql_rfm, x_val, y_val, z_val, color='ClusterLabels', labels='ClusterLabels', title="<b> RFM KMeans Clustering Customer Distribution</b>", color_continuous_scale='turbid')

            def kmeans_level(df):
                if (df['ClusterLabels'] == 0):
                    return 'Promising'
                elif (df['ClusterLabels'] == 1):
                    return 'Needs Attention'
                elif (df['ClusterLabels'] == 2):
                    return 'Loyal'
            dfsql_rfm['RFM with KMeans Level'] = dfsql_rfm.apply(kmeans_level, axis=1)

            loyal_kmeans = dfsql_rfm['RFM with KMeans Level'].value_counts()['Loyal']
            promising_kmeans = dfsql_rfm['RFM with KMeans Level'].value_counts()['Promising']
            need_attention_kmeans = dfsql_rfm['RFM with KMeans Level'].value_counts()['Needs Attention']

            st.subheader("RFM (Recency, Frequency, Monetary) with KMeans Clustering:")
            st.caption(f"Total Customer: {dfsql_rfm['CustomerID'].count()}")
            left_column, middle_column, right_column = st.columns(3)
            with left_column:
                st.caption("Loyal:")
                st.caption(f"{loyal_kmeans}")
            with middle_column:
                st.caption("Promising")
                st.caption(f"{promising_kmeans}")
            with right_column:
                st.caption("Need Attention:")
                st.caption(f"{need_attention_kmeans}")

            fig_rfm_score_pie = px.pie(dfsql_rfm, values=[loyal_kmeans, promising_kmeans, need_attention_kmeans], names=["Loyal", "Promising", "Need Attention"], title="<b>RFM with KMeans Level Distribution</b>")
        
            left_column, right_column = st.columns(2)
            
            left_column.plotly_chart(fig_rfm_kmeans_clustering_3d, use_container_width=True)

            right_column.plotly_chart(fig_rfm_score_pie, use_container_width=True)
            return dfsql_rfm
        dfsql_rfm = runModelDevelopment(dfsql_rfm)

        def filter_dataframe(df: pd.DataFrame, key_name: str) -> pd.DataFrame:
            modify = st.checkbox("Add filters", key=f"{key_name}")
            if not modify:
                return df
            df = df.copy()
            for col in df.columns:
                if is_object_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass
                if is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.tz_localize(None)
            modification_container = st.container()
            with modification_container:
                to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
                for column in to_filter_columns:
                    left, right = st.columns((1, 20))
                    if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                        user_cat_input = right.multiselect(f"Values for {column}",df[column].unique(),default=list(df[column].unique()),)
                        df = df[df[column].isin(user_cat_input)]
                    elif is_numeric_dtype(df[column]):
                        _min = float(df[column].min())
                        _max = float(df[column].max())
                        step = (_max - _min) / 100
                        user_num_input = right.slider(f"Values for {column}",min_value=_min,max_value=_max,value=(_min, _max),step=step,)
                        df = df[df[column].between(*user_num_input)]
                    elif is_datetime64_any_dtype(df[column]):
                        user_date_input = right.date_input(f"Values for {column}",value=(df[column].min(),df[column].max(),),)
                        if len(user_date_input) == 2:
                            user_date_input = tuple(map(pd.to_datetime, user_date_input))
                            start_date, end_date = user_date_input
                            df = df.loc[df[column].between(start_date, end_date)]
                    else:
                        user_text_input = right.text_input(f"Substring or regex in {column}",)
                        if user_text_input:
                            df = df[df[column].astype(str).str.contains(user_text_input)]

            return df
        st.dataframe(filter_dataframe(dfsql_rfm, "kmeans"), use_container_width=True)

        # @st.cache_data
        # def runDownloadDataframe(dfsql_rfm, dfsql_selection):
            
        #     # dfsql_rfm_raw = dfsql_rfm.merge(dfsql_selection[['CustomerID', 'UnitPrice', 'Product']], on='CustomerID', how='left')
        #     # st.download_button('Download Dataframe of RFM result as CSV', dfsql_rfm.to_string(), 'rfm_result_dataframe.csv', 'text/csv')

        # runDownloadDataframe(dfsql_rfm, dfsql_selection)
        # @st.cache_data
        # def convert_df_rfm_result(df):
        #     return df.to_csv().encode('utf-8')
        # csv = convert_df_rfm_result(dfsql_rfm)
        # st.download_button('Download Dataframe of RFM result as CSV', csv, 'rfm_result_dataframe.csv', 'text/csv')
       

    #     print("Max tanggal", dfsql['InvoiceDate'].max())
        

    #     print("Dataframe from sql Monetary: ", dfsql["Monetary"])
    
    # #-----------------------------RECENCY FREQUENCY MONETARY SCORE-----------------------------
    #     dfsql_rfm['recency_val'] = pd.qcut(dfsql_rfm['Recency'].rank(method="first"), q=3, labels=['1','2','3'])
    #     dfsql_rfm['frequency_val'] = pd.qcut(dfsql_rfm['Frequency'].rank(method="first"), q=3, labels=['3','2','1'])
    #     dfsql_rfm['monetary_val'] = pd.qcut(dfsql_rfm['Monetary'].rank(method="first"), q=3, labels=['3','2','1'])
    #     dfsql_rfm['RFM Group'] = dfsql_rfm.recency_val.astype(str) + dfsql_rfm.frequency_val.astype(str)
    #     dfsql_rfm['RFM Score'] = dfsql_rfm[['recency_val', 'frequency_val', 'monetary_val']].astype(int).sum(axis=1)
        
    #     dfsql_rfm = dfsql_rfm.sort_values(by="RFM Score", ascending=True).reset_index(drop=True)

        
    #     def rfm_level(df):
    #         if (df['RFM Score'] == 9):
    #             return 'Needs Attention'
    #         elif ((df['RFM Score'] >= 6) and (df['RFM Score'] <= 8)):
    #             return 'Promising'
    #         elif ((df['RFM Score'] >= 3) and (df['RFM Score'] <= 5)):
    #             return 'Loyal'
    #     dfsql_rfm['RFM Level'] = dfsql_rfm.apply(rfm_level, axis=1)




    #     loyal = dfsql_rfm['RFM Level'].value_counts()['Loyal']
    #     promising = dfsql_rfm['RFM Level'].value_counts()['Promising']
    #     need_attention = dfsql_rfm['RFM Level'].value_counts()['Needs Attention']

    #     print("RFM with Level loyal:", loyal)          
    #     print("Promising Customer : ", promising)
    #     print("Need Attention Customer : ", need_attention)
    #     # average_rating = round(df_selection["Rating"].mean(), 1)
    #     # start_rating = ":star:" * int(round(average_rating, 0))
    #     average_sale_by_transaction = round((dfsql_selection["UnitPrice"]*dfsql_selection["Quantity"]).mean(), 2)

    #     st.subheader("RFM (Recency, Frequency, Monetary):")
    #     #show caption total customer and
    #     st.caption(f"Total Customer: {dfsql_rfm['CustomerID'].count()}")
    #     left_column, middle_column, right_column = st.columns(3)
    #     with left_column:
    #         st.caption("Loyal:")
    #         st.caption(f"{loyal}")
    #     with middle_column:
    #         st.caption("Promising")
    #         st.caption(f"{promising}")
    #     with right_column:
    #         st.caption("Need Attention:")
    #         st.caption(f"{need_attention}")


    #     #-----------------------------SCATTER 3D-----------------------------
    #     x_val = 'Recency'
    #     y_val = 'Frequency'
    #     z_val = 'Monetary'

    #     fig_rfm_score_3d = px.scatter_3d(dfsql_rfm, x_val, y_val, z_val, color='RFM Level', labels='RFM Level', title="<b> RFM Customer Distribution</b>", color_discrete_sequence=['#5d4232','#a8773c','#dfe292'])
    #     # st.plotly_chart(fig_rfm_score_3d, use_container_width=True)
    #     # fig_rfm_score_3d.update_layout(color_continuous_scale='Viridis')
                
    #     #-----------------------------PIE CHART-----------------------------
    #     fig_rfm_score_pie = px.pie(dfsql_rfm, values=[loyal, promising, need_attention], names=["Loyal", "Promising", "Need Attention"], title="<b> RFM Customer Level Distribution</b>")

    #     left_column, right_column = st.columns(2)
        
    #     left_column.plotly_chart(fig_rfm_score_3d, use_container_width=True)

    #     right_column.plotly_chart(fig_rfm_score_pie, use_container_width=True)       

    #     # st.plotly_chart(fig_rfm_clustering_3d, use_container_width=True)
        
    #     # show dataframe rfm
    #     st.dataframe(filter_dataframe(dfsql_rfm, "rfm"), use_container_width=True)


        #-----------------------------CLUSTERING K-MEANS-----------------------------
        
        


        #-----------------------------CLUSTERING K-MEDOIDS-----------------------------



        #-----------------------------SUMMARY JUMLAH TRANSAKSI, PELANGGAN, DAN SALES [LINE CHART]-----------------------------
        # summary_by_month = dfsql_selection.groupby(['bulan']).agg({'JumlahPelanggan': 'count', 'JumlahTransaksi': 'nunique', 'sales': 'sum'}).reset_index()
        # print("Summary Per Bulan:", summary_by_month)


        


        # show scatter plot
        # left_column, right_column = st.columns(2)
        
        # left_column.plotly_chart(fig_rfm_score_3d, use_container_width=True)

        # right_column.plotly_chart(fig_rfm_clustering_3d, use_container_width=True)



        #HIDE CERTAIN STREAMLIT STYLE USING CUTOM CSS
    else:
        st.header("Data is empty")



    streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Roboto', sans-serif;
			}
			</style>
			"""
    st.markdown(streamlit_style, unsafe_allow_html=True)


    # hide_st_style = """
    #                 <style>
    #                 #MainMenu {visibility: hidden;}
    #                 footer {visibility: hidden;}
    #                 header {visibility: hidden;}
    #                 .row_heading.level0 {display:none}
    #                 .blank {display:none}
    #                 </style>
    #                 """
    # st.markdown(hide_st_style, unsafe_allow_html=True)

    # hide_streamlit_style = """
    #         <style>
    #         #MainMenu {visibility: hidden;}
    #         footer {visibility: hidden;}
    #         </style>
    #         """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    # hide_footer_style = """
    #     <style>
    #     .reportview-container .main footer {visibility: hidden;}    
    #     """
    # st.markdown(hide_footer_style, unsafe_allow_html=True)





# KOMEN DHULU

# # ----USER-AUTH
# names = ["Admin 1", "Admin 2"]
# usernames = ["admin1", "admin2"]
# file_path = Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open("rb") as file:
#     hashed_passwords = pickle.load(file)
    

# # users =  database.

# credentials = {
#         "usernames":{
#             usernames[0]:{
#                 "name":names[0],
#                 "password":hashed_passwords[0]
#                 },
#             usernames[1]:{
#                 "name":names[1],
#                 "password":hashed_passwords[1]
#                 }            
#             }
#         }

# authenticator = streamlit_authenticator.Authenticate(credentials, "sales_dashboard", "abcdef", cookie_expiry_days=30)


# usernames = dict(user[0] for user in users)
# hashed_passwords = dict(user[1] for user in users)
# names = dict(user[2] for user in users)


# bar pemanggilan filterDashboard()

   # dfsql_selection_by_all = dfsql_selection_by_country.query(
    #     "Country == @country & tahun == @tahun & bulan == @bulan"
    # )
    
    
    # ??if using range
    # if len(a_date) == 2:
        # mask = (dfsql['InvoiceDate'] > a_date[0].strftime("%Y-%m-%d")) & (dfsql['InvoiceDate'] <= a_date[1].strftime("%Y-%m-%d"))
        # dfsql_selection1 = dfsql.loc[mask]
        # country = st.sidebar.multiselect(
        #     "Select the City:",
        #     options=dfsql_selection1["Country"].unique(),
        #     default=dfsql_selection1["Country"].unique()
        # )        
    #     dfsql_selection = dfsql_selection1.query(
    #         "Country == @country"
    #     )
    # else:
    #     dfsql_selection = dfsql
