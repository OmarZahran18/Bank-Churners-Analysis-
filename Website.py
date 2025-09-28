import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churners Dashboard", layout="wide")

st.title("📊 Bank Churners Analysis Dashboard")
st.sidebar.header("🔎 Navigation")
st.sidebar.markdown("Created by [Omar Zahran](https://www.linkedin.com/in/omarzahran22/)")

@st.cache_data
def load_data():
    data = pd.read_csv(r"D:\Power Bi @ITI\Python Visualization\Project\BankChurners.csv")
    data = data.drop([
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
    ], axis=1)
    data.rename(columns={
        'CLIENTNUM': 'Client_Number',
        'Dependent_count': 'Dependents',
        'Months_Inactive_12_mon': 'Inactive_Months',
        'Contacts_Count_12_mon': 'Contacts_Counts'
    }, inplace=True)
    return data

data = load_data()

option = st.sidebar.radio("Go to:", ["📋 Data Overview", "📈 EDA & Visualization"])

# ---------------- DATA OVERVIEW ----------------
if option == " Data Overview":
    st.subheader(" Dataset Preview")
    st.write(data.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(data):,}")
    col2.metric("Existing Customers", f"{(data['Attrition_Flag']=='Existing Customer').sum():,}")
    col3.metric("Attrited Customers", f"{(data['Attrition_Flag']=='Attrited Customer').sum():,}")

    st.write("###  Shape:")
    st.write(data.shape)

    st.write("###  Summary Statistics:")
    st.write(data.describe())

# ---------------- EDA & VISUALIZATION ----------------
elif option == " EDA & Visualization":
    st.subheader("Exploratory Data Analysis & Visualizations")

    # ---------------- KPI CARDS ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Age", round(data['Customer_Age'].mean(), 1))
    col2.metric("Avg Credit Limit", f"${data['Credit_Limit'].mean():,.0f}")
    col3.metric("Avg Transactions", f"{data['Total_Trans_Ct'].mean():.0f}")
    col4.metric("Avg Transaction Amount", f"${data['Total_Trans_Amt'].mean():,.0f}")

    # ---------------- Gender Pie ----------------
    st.write("### of Customers by Gender")
    fig = px.pie(data, names='Gender', title="Gender Distribution",
                 color='Gender', color_discrete_map={'F':'darkorange','M':'darkblue'})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Education Level ----------------
    st.write("### 🎓 Number of Customers by Education Level")
    edu_counts = data['Education_Level'].value_counts().reset_index()
    edu_counts.columns = ['Education_Level','Count']
    fig = px.bar(edu_counts, x='Education_Level', y='Count',
                 labels={'Education_Level':'Education Level','Count':'Number of Customers'},
                 color='Count', color_continuous_scale='Viridis')
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Marital Status ----------------
    st.write("### 💍 Number of Customers by Marital Status")
    marital_counts = data['Marital_Status'].value_counts().reset_index()
    marital_counts.columns = ['Marital_Status','Count']
    fig = px.bar(marital_counts, x='Marital_Status', y='Count',
                 labels={'Marital_Status':'Marital Status','Count':'Number of Customers'},
                 color='Count', color_continuous_scale='Cividis')
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Income Category ----------------
    st.write("### 💰 Number of Customers by Income Category")
    income_counts = data['Income_Category'].value_counts().reset_index()
    income_counts.columns = ['Income_Category','Count']
    fig = px.bar(income_counts, x='Income_Category', y='Count',
                 labels={'Income_Category':'Income Category','Count':'Number of Customers'},
                 color='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Card Category ----------------
    st.write("### 💳 Number of Customers by Card Category")
    card_counts = data['Card_Category'].value_counts().reset_index()
    card_counts.columns = ['Card_Category','Count']
    fig = px.bar(card_counts, x='Card_Category', y='Count',
                 labels={'Card_Category':'Card Category','Count':'Number of Customers'},
                 color='Count', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Attrition by Card Category ----------------
    st.write("### 🔄 Attrition Distribution by Card Category")
    card_attr = pd.crosstab(data['Card_Category'], data['Attrition_Flag'], normalize='index') * 100
    card_attr = card_attr.reset_index().melt(id_vars='Card_Category', var_name='Attrition_Flag', value_name='Percentage')
    fig = px.bar(card_attr, 
                 x='Card_Category', 
                 y='Percentage', 
                 color='Attrition_Flag', 
                 barmode='stack',
                 text='Percentage',
                 color_discrete_map={'Existing Customer':'darkgreen','Attrited Customer':'red'},
                 title="Attrition by Card Category (%)")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Numeric Distributions ----------------
    st.write("### 📈 Distribution of Numeric Columns")
    numeric_cols = ['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct']
    for col in numeric_cols:
        fig = px.histogram(data, x=col, nbins=30, title=f'Distribution of {col}', marginal="box",
                           color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Attrition Pie ----------------
    st.write("### 🧾 Customer Attrition Distribution")
    fig = px.pie(data, names='Attrition_Flag', color='Attrition_Flag', 
                 color_discrete_map={'Existing Customer':'darkgreen','Attrited Customer':'red'})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Attrition by Gender ----------------
    st.write("### 👨‍🦱👩 Attrition by Gender")
    fig = px.histogram(data, x='Gender', color='Attrition_Flag', barmode='group',
                       color_discrete_map={'Existing Customer':'darkgreen','Attrited Customer':'red'})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Correlation Heatmap ----------------
    st.write("### 🔥 Correlation Heatmap")
    corr = data[['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct',
                 'Avg_Open_To_Buy','Inactive_Months']].corr()

    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
