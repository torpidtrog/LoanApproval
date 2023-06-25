import streamlit as st
import pandas as pd
import plotly.express as px
import random
import functions
import seaborn as sns
import matplotlib.pyplot as plt  

st.set_page_config(
    layout = "wide", 
    page_icon = 'logo.png',
    page_title='EDA')

st.header("Exploratory Data Analysis")

functions.space()

dataset = 'loan_train.csv'

if dataset:
    df = pd.read_csv(dataset)
    df['Result']= df.apply(
    lambda row: 'Good' if row['Label'] == 0 else 'Bad', axis=1
    )
    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns
    with st.expander("See data", False):
        df

    all_visuals = ['Descriptive Analysis', 
                   'Correlation Plot',
                   'Distribution of Numerical Columns', 
                   'Count Plots of Categorical Columns', 
                   'Box Plots', 
                   'Outlier Analysis']
    functions.sidebar_space(3)         
    visuals = st.sidebar.selectbox("Pick from the list below:", all_visuals)

    if 'Descriptive Analysis' in visuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())
    elif 'Correlation Plot' in visuals:
        corr_plot_data = df.loc[:, df.columns!='Result']
        c_columns = corr_plot_data.columns
        if "default_ms" not in st.session_state:
            st.session_state["default_ms"] = random.sample(list(c_columns), 5)
        s_columns = st.sidebar.multiselect('Which columns would you like to use?', list(c_columns), st.session_state["default_ms"])
        if len(s_columns)==0:
            st.write("")
        else:
            corr_matrix = corr_plot_data[s_columns].corr()
            fig,ax = plt.subplots(
                figsize=(10,5)
                )
            sns.heatmap(
                corr_matrix.corr(), 
                annot = True, 
                cmap = "coolwarm",
                ax = ax
                )
            fig.savefig('corrplot.png', format="png")
            st.image('corrplot.png')
    elif 'Distribution of Numerical Columns' in visuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    elif 'Count Plots of Categorical Columns' in visuals:

        if len(cat_columns) == 0:
            st.write('There is no categorical columns in the data.')
        else:
            selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_cat_cols)):
                        break

                    fig = px.histogram(df, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1

    elif 'Box Plots' in visuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    
                    if (i >= len(selected_num_cols)):
                        break
                    
                    fig = px.box(df, y = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    elif 'Outlier Analysis' in visuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions.number_of_outliers(df))

