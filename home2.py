import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


st.set_page_config(
    page_title="Social Vitals",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "### Created By *Kajal, 2541170*"
    }
)

@st.cache_data
def load_data():
    return pd.read_csv("social_media_impact_on_health_cleaned.csv")

df = load_data()
# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("📱 Social Care")
st.sidebar.subheader("Created by : Kajal | 2541170")
st.sidebar.divider()
menu = st.sidebar.radio(
    "Navigation Links",
    (
        "🏠 Home & Dataset Overview",
        "🧹 Data Cleaning",
        "📊 Exploratory Data Analysis",
        "🔍 Research Questions",
        "🔮 Prediction", 
        "📌 Research Questions Summary",
        "📌 Conclusion"
    )
)


# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------
if menu == "🏠 Home & Dataset Overview":
    st.markdown("""
    # 📱 Social Vitals
    ### *Social Media Usage and Health Analysis Dashboard*
    """)
    st.markdown("---")

    # Project Description
    st.subheader("📌 Project Description")

    st.info("""
    **Social Vitals** is a data analytics and visualization project that examines how social media usage
    influences students' health and well-being. The dashboard provides interactive visualizations
    and insights into the relationship between daily social media usage, mental health, sleep duration,
    academic performance, age, and gender.

    The project is developed using **Python, Streamlit, Pandas, and Plotly** to transform raw data
    into meaningful insights that support better understanding and decision-making.
    """)

    st.subheader("📖 Project Overview")
    st.info("""
    The Social Vitals dashboard provides a comprehensive analysis of students'
    social media usage and its influence on overall well-being. Users can
    explore the dataset through interactive tables, charts, and summary
    statistics to identify trends and relationships between social media
    usage, sleep duration, mental health, academic performance, age, gender,
    academic level, and country.
    """)
    
    st.subheader("Project Objectives")
    st.markdown("""
    - Analyze students' social media usage patterns.
    - Study the relationship between social media usage and mental health.
    - Examine the effect of social media on sleep duration.
    - Explore the impact on academic performance.
    - Compare social media usage across age groups and gender.
    - Present interactive visualizations for better understanding.
    """)

    # Key Features
    st.subheader("✨ Key Features")

    st.success("""
    ✔ Interactive and user-friendly dashboard

    ✔ Dataset preview and statistical summary

    ✔ Data cleaning and preprocessing analysis

    ✔ Exploratory Data Analysis (EDA)

    ✔ Research question analysis

    ✔ Interactive Plotly visualizations

    ✔ Insights into students' health and social media usage
    """)

    
    st.markdown("---")
   
    # Dataset Information
    st.subheader("📈 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Dataset Description
    st.subheader("📊 Dataset Description")

    st.success("""
    The dataset contains information about students' social media usage and its impact on health.
    It includes demographic details, social media habits, sleep patterns, mental health scores,
    academic performance, and the most frequently used social media platforms.

    The dataset is used to perform Exploratory Data Analysis (EDA), visualize trends,
    and identify relationships between social media usage and health outcomes.
    """)

    st.markdown("### 📋 Dataset Features")

    st.write("""
    The dataset contains information about students' social media usage and its impact on health and
    well-being. It includes demographic information, daily usage patterns, sleep duration,
    mental health scores, academic performance, and the most frequently used social media platforms.
    """)

    feature_info = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Mental_Health_Score",
            "Affects_Academic_Performance",
            "Overall_Impact",
            "Most_Used_Platform",
            "Academic_Level",
            "Country"
        ],
        "Description": [
            "Age of the student",
            "Gender of the student",
            "Average daily social media usage (hours)",
            "Average sleep hours per night",
            "Mental health score",
            "Whether social media affects academic performance",
            "Overall impact of social media",
            "Most frequently used social media platform",
            "Current academic level",
            "Country of the student"
        ]
    })

    st.dataframe(feature_info, width="stretch")

    # Display Dataset
    st.subheader("📄 Dataset Preview")
    st.dataframe(df, width="stretch")

    # Data Types
    st.subheader("📋 Data Types")

    dtype_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(dtype_df, width="stretch")

    st.markdown("---")

    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    st.write("""
    The statistical summary provides descriptive statistics for numerical and
    categorical variables, including count, mean, standard deviation,
    minimum, maximum, and quartile values.
    """)

    st.dataframe(df.describe(include="all"), width="stretch")

    st.markdown("---")

    st.subheader("📝 Key Highlights")

    st.success("""
    ✔ Dataset contains student demographic and social media usage information.

    ✔ Includes health-related variables such as sleep duration and mental health score.

    ✔ Useful for Exploratory Data Analysis (EDA), visualization, and predictive analysis.

    ✔ Supports identifying relationships between social media usage and student well-being.
    """)  

# ----------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------
elif menu == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning")
    st.markdown("---")

    st.write("""
    Data cleaning is an important step in data analysis. It helps improve the
    quality of the dataset by identifying and handling missing values,
    duplicate records, and incorrect data types.
    """)

    # Original Dataset
    st.subheader("📄 Original Dataset")
    st.dataframe(df, width="stretch")

    st.markdown("---")

    # Dataset Shape
    st.subheader("📏 Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.markdown("---")

    # Missing Values
    st.subheader("❌ Missing Values")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(missing, width="stretch")

    if missing["Missing Values"].sum() == 0:
        st.success("✅ No missing values found in the dataset.")
    else:
        st.warning("⚠ Missing values detected.")

    st.markdown("---")

    # Duplicate Records
    st.subheader("🔄 Duplicate Records")

    duplicates = df.duplicated().sum()

    st.metric("Duplicate Rows", duplicates)

    if duplicates == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠ {duplicates} duplicate rows found.")

    st.markdown("---")

    # Data Types
    st.subheader("📋 Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(dtype_df, width="stretch")

    st.markdown("---")

    # Cleaning Process
    st.subheader("⚙ Data Cleaning Process")

    cleaned_df = df.copy()

    # Remove duplicate rows
    cleaned_df.drop_duplicates(inplace=True)

    # Fill missing values
    import pandas as pd

    for col in cleaned_df.columns:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        else:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    st.success("✅ Duplicate rows removed.")
    st.success("✅ Missing values handled.")

    st.markdown("---")

    # Cleaned Dataset
    st.subheader("🧹 Cleaned Dataset")

    st.dataframe(cleaned_df, width="stretch")

    st.markdown("---")

    # Cleaned Dataset Summary
    st.subheader("📊 Cleaned Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", cleaned_df.shape[0])

    with col2:
        st.metric("Columns", cleaned_df.shape[1])

    with col3:
        st.metric("Missing Values", cleaned_df.isnull().sum().sum())

    st.markdown("---")

   
    
# ----------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------
elif menu == "📊 Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    st.write("""
    Exploratory Data Analysis (EDA) helps understand the dataset by
    identifying patterns, trends, and relationships through interactive
    visualizations.
    """)
    required_columns = [
        "Avg_Daily_Usage_Hours",
        "Gender",
        "Age",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Overall_Impact",
        "Most_Used_Platform",
        "Affects_Academic_Performance",
        "Country"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # ----------------------------------------------------  
    # 1 Average Daily Usage Distribution
    # ----------------------------------------------------
    st.subheader("1️⃣ Average Daily Social Media Usage Distribution")

    fig = px.histogram(
        df,
        x="Avg_Daily_Usage_Hours",
        nbins=20,
        title="Distribution of Daily Social Media Usage",
        color_discrete_sequence=["#4F81BD"]
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5
    )

    st.plotly_chart(fig, width="stretch")

    st.info("""
    ### 📊 Analysis

    - This histogram shows the distribution of students based on their average daily social media usage.
    - **X-axis:** Average Daily Social Media Usage (Hours).
    - **Y-axis:** Number of Students (Frequency).
    - Each bar represents the number of students whose daily usage falls within a particular range of hours.
    - The chart indicates how social media usage is distributed across the student population.
    """)

    st.success("""
    ### 💡 Insights

    - The highest concentration of students is around **4.5–6.0 hours** of daily social media usage.
    - Most students spend approximately **4–7 hours per day** on social media.
    - Very few students spend **less than 2 hours** or **more than 8 hours** per day.
    - The distribution is concentrated around the middle usage ranges, indicating that moderate social media usage is more common than extremely low or extremely high usage.
    - The chart shows a gradual decrease in the number of students as daily usage moves away from the central range.
    """)

    st.markdown("---")

    # ----------------------------------------------------
    # 2 Gender Distribution
    # ----------------------------------------------------
    st.subheader("2️⃣ Gender Distribution")

    gender = df["Gender"].value_counts().reset_index()
    gender.columns = ["Gender", "Count"]

    fig = px.pie(
    gender,
    names="Gender",
    values="Count",
    hole=0.5,
    color_discrete_sequence=["#4F81BD","#7EA6E0"]
)

    fig.update_layout(
        template="plotly_dark",
        title="Gender Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This donut chart illustrates the gender distribution of students in the dataset.
    - Each slice represents the proportion of students belonging to a specific gender.
    - The percentage displayed on each slice indicates the share of that gender in the overall dataset.
    - The chart helps assess whether the dataset is balanced in terms of gender representation.
    """)

    st.success("""
    ### 💡 Insights

    - Male students account for **51.5%** of the dataset, making them the largest group.
    - Female students represent **48.5%** of the dataset.
    - The difference between male and female representation is **only 3%**, indicating a nearly balanced gender distribution.
    - Since both genders are almost equally represented, the dataset is well-balanced and suitable for unbiased gender-based comparisons and analysis.
    """)

    st.markdown("---")

    #--------------------------------------------------------
    # 3 Age Distribution
    # -------------------------------------------------------
    st.subheader("3️⃣ Age Distribution")

    age = df["Age"].value_counts().sort_index().reset_index()
    age.columns = ["Age", "Count"]

    fig = px.bar(
        age,
        x="Age",
        y="Count",
        color_discrete_sequence=["#4F81BD"]
    )

    fig.update_layout(
        template="plotly_dark",
        title="Age Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")
    st.info("""
    ### 📊 Analysis

    - This bar chart shows the distribution of students across different age groups.
    - **X-axis:** Students' Age (18–24 years).
    - **Y-axis:** Number of Students (Count).
    - Each bar represents the total number of students belonging to a particular age group.
    - The chart helps identify which age groups are most and least represented in the dataset.
    """)

    st.success("""
    ### 💡 Insights

    - Students aged **21 years** have the highest representation in the dataset.
    - The **19, 20, and 22-year-old** groups also have a high number of students, with similar counts.
    - Students aged **18 and 24 years** have the lowest representation.
    - The majority of students fall within the **19–22 years** age range, indicating that the dataset mainly consists of young adults in typical college-age groups.
    - The lower representation of ages **18, 23, and 24** suggests these age groups are comparatively less common in the dataset.
    """)
    st.markdown("---")

    # ----------------------------------------------------
    # 4 Sleep Hours
    # ----------------------------------------------------
    st.subheader("4️⃣ Sleep Hours Distribution")

    fig = px.box(
        df,
        x="Gender",
        y="Sleep_Hours_Per_Night",
        color="Gender",
        color_discrete_sequence=["#4F81BD","#7EA6E0"]
)

    fig.update_layout(
        template="plotly_dark",
        title="Sleep Hours per Night",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This box plot compares the distribution of sleep hours per night between male and female students.
    - Both genders have a similar median sleep duration of around 6.5 hours.
    - The spread of sleep hours is almost the same for both male and female students.
    - Most students sleep between approximately 5.5 and 7.5 hours per night.
    - There are no major differences in sleep patterns between the two groups.
    """)

    st.success("""
    ### 💡 Insights

    - Male and female students have very similar sleep habits.

    - The average sleep duration is approximately 6–7 hours per night for both groups.

    - Sleep patterns are consistent across genders, with only minor variations.

    - No significant differences or unusual outliers are observed in the distribution.

    - The results suggest that gender has little influence on students' sleep duration in this dataset.
    """)
    st.markdown("---")

    # ----------------------------------------------------
    # 5 Mental Health
    # ----------------------------------------------------
    st.subheader("5️⃣ Mental Health Score Distribution")

    fig = px.histogram(
        df,
        x="Mental_Health_Score",
        nbins=20,
        title="Mental Health Score Distribution",
        color_discrete_sequence=["#4F81BD"]
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This histogram shows the distribution of students' **Mental Health Scores**.
    - **X-axis:** Mental Health Score.
    - **Y-axis:** Number of Students (Frequency).
    - Each bar represents the number of students whose mental health scores fall within a specific score range.
    - The chart helps identify the most common mental health score ranges and the overall spread of scores in the dataset.
    """)

    st.success("""
    ### 💡 Insights

    - The highest concentration of students has mental health scores between **5 and 7**, with the peak occurring around **6**.
    - Scores below **5** and above **8** are comparatively less common, indicating fewer students at the extreme ends of the scale.
    - The distribution suggests that most students have **moderate mental health scores**, while very low and very high scores are relatively rare.
    - The spread of scores indicates some variation in mental well-being, but the majority of students are clustered around the middle score range.
    - Overall, the dataset reflects that students generally report **average to moderately good mental health** rather than extremely low or extremely high mental health scores.
    """)

    st.markdown("---")

    # ----------------------------------------------------
    # 6 Overall Impact
    # ----------------------------------------------------
    st.subheader("6️⃣ Overall Impact")

    impact = df["Overall_Impact"].value_counts().reset_index()
    impact.columns = ["Impact", "Count"]

    fig = px.pie(
    impact,
    names="Impact",
    values="Count",
    color_discrete_sequence=["#4F81BD","#7EA6E0","#A9C4EB"]
)

    fig.update_layout(
        template="plotly_dark",
        title="Overall Impact",
        title_x=0.5
    )


    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This pie chart illustrates the overall impact of social media on students.
    - Each slice represents one impact category (Positive, Negative, or Neutral).
    - The size of each slice indicates the proportion of students in that category.
    - The chart provides an overview of students' perceptions of social media's impact.
    """)

    st.success("""
    ### 💡 Insights

    - The **Negative** category represents the largest proportion of students, indicating that many students perceive social media to have a negative impact.
    - The **Positive** category is the second largest, showing that a considerable number of students also experience beneficial effects.
    - The **Neutral** category has the smallest proportion, suggesting that relatively few students feel social media has neither a positive nor a negative impact.
    - Overall, the chart suggests that negative perceptions of social media are more common than positive or neutral perceptions among students.
    """)

    st.markdown("---")

    # ----------------------------------------------------
    # 7 Most Used Platform
    # ----------------------------------------------------
    st.subheader("7️⃣ Most Used Social Media Platform")

    platform = df["Most_Used_Platform"].value_counts().reset_index()
    platform.columns = ["Platform", "Users"]


    fig = px.bar(
        platform,
        x="Platform",
        y="Users",
        color_discrete_sequence=["#4F81BD"]
    )

    fig.update_layout(
        template="plotly_dark",
        title="Most Used Social Media Platforms",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This bar chart visualizes the popularity of different social media platforms among students.
    - Each bar represents a social media platform.
    - The height of each bar corresponds to the number of students using that platform.
    - Larger bars indicate more popular platforms, while smaller bars represent platforms with fewer users.
    - The chart enables quick comparison of platform popularity within the dataset.
    """)

    st.success("""
    ### 💡 Insights

    - **Instagram** is the most widely used social media platform, having the largest share of users.
    - **TikTok** and **Facebook** are the next most popular platforms, indicating strong student engagement on these platforms.
    - **LinkedIn**, **Twitter**, **YouTube**, and **Snapchat** have moderate usage compared to the top three platforms.
    - **WhatsApp**, **WeChat**, **LINE**, **KakaoTalk**, and **Weibo** have the smallest user base, making them the least preferred platforms in this dataset.
    - Overall, students show a stronger preference for entertainment and content-sharing platforms rather than messaging or professional networking platforms.
    """)

    st.markdown("---")

    # -----------------------------------------------------
    # 8 Academic Performance
    # -----------------------------------------------------
    st.subheader("8️⃣ Academic Performance")

    academic = df["Affects_Academic_Performance"].value_counts().reset_index()
    academic.columns = ["Performance", "Count"]

    fig = px.bar(
        academic,
        x="Performance",
        y="Count",
        color_discrete_sequence=["#4F81BD"]
    )

    fig.update_layout(
        template="plotly_dark",
        title="Academic Performance Impact",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This choropleth map shows the geographical distribution of students across different countries.
    - Darker shades represent countries with a higher number of students.
    - Hovering over a country displays its student count.
    - The map helps identify regions with higher student participation.
    """)

    st.success("""
    ### 💡 Insights

    - Countries with darker colors have the highest number of students.
    - Lighter-colored countries have fewer student records.
    - Student participation is unevenly distributed across countries.
    - The map provides a clear geographical overview of the dataset and highlights countries with greater representation.
    """)

    st.markdown("---")

    # -----------------------------------------------------
    # 9 Country Distribution
    # -----------------------------------------------------
    st.subheader("9️⃣ Country Distribution")

    # Count students by country
    country_counts = df["Country"].value_counts().reset_index()

    # Rename columns
    country_counts.columns = ["Country", "Students"]


    fig = px.choropleth(
        country_counts,
        locations="Country",
        locationmode="country names",
        color="Students",
        hover_name="Country",
        hover_data=["Students"],
        color_continuous_scale="Blues",
        title="Country-wise Student Distribution"
    )

    # Layout
    fig.update_layout(
        template="plotly_white",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            bgcolor="Blue"
        )
    )

    st.plotly_chart(fig, width="stretch")

    st.info("""
    ### 📊 Analysis

    - The choropleth map displays the distribution of students across different countries.
    - Darker blue shades represent countries with a higher number of students, while lighter shades indicate fewer students.
    - Student representation is concentrated in a limited number of countries.
    - Several countries appear white, indicating no student records or missing data.
    - The map provides a clear geographical overview of where students are distributed worldwide.
    """)
        
    st.success("""
    ### 💡 Insights

    - Student distribution is uneven, with a few countries contributing the majority of students.

    - Countries highlighted with darker blue colors represent the largest student populations.

    - Many regions have little or no student representation, suggesting opportunities for future outreach.

    - The visualization helps identify key countries for student recruitment and international engagement.

    -  Geographic analysis enables institutions to make data-driven decisions for global expansion and resource planning.
    """)

    st.markdown("---")

    # -----------------------------------------------------
    # 10 Correlation Heatmap
    # -----------------------------------------------------
    st.subheader("🔟 Correlation Heatmap")
  


    fig.update_layout(
        template="plotly_dark",
        title="Correlation Heatmap",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.info("""
    ### 📊 Analysis

    - This correlation heast.subheader("🔟 Correlation Heatmap")

    corr = df.select_dtypes(include="number").corr()
    print("Number Columns :- ", corr)
    
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Blues",tmap shows the relationships between the numerical variables in the dataset.
    - Each cell displays the correlation coefficient between two variables, ranging from **-1 to +1**.
    - Positive values indicate that two variables increase together, while negative values indicate an inverse relationship.
    - Values close to **0** suggest little or no linear relationship between the variables.
    - Darker colors represent stronger correlations, whereas lighter colors indicate weaker correlations.
    """)

    st.success("""
    ### 💡 Insights

    - **Average Daily Social Media Usage** has a **strong negative correlation (-0.82)** with **Sleep Hours Per Night**, indicating that students who spend more time on social media tend to sleep fewer hours.
    - **Average Daily Social Media Usage** also shows a **strong negative correlation (-0.83)** with **Mental Health Score**, suggesting that increased social media usage is associated with lower mental health scores.
    - **Sleep Hours Per Night** has a **strong positive correlation (0.79)** with **Mental Health Score**, meaning students who sleep more generally report better mental health.
    - **Age** has a very weak correlation with the other variables, indicating that age has little influence on social media usage, sleep duration, or mental health in this dataset.
    - **Student_ID** shows negligible correlations with all variables, confirming that it is only an identifier and does not contribute to meaningful analysis.
    - Overall, the heatmap suggests that **higher social media usage is associated with reduced sleep and poorer mental health, while better sleep is linked to improved mental health.**
    """)

# ----------------------------------------------------
# 🔍 RESEARCH QUESTIONS
# ----------------------------------------------------

elif menu == "🔍 Research Questions":

    st.title("🔍 Research Questions")

    st.markdown("---")

    st.write("""
    This section answers the research questions using interactive Plotly
    visualizations along with detailed analysis and insights.
    """)

    # ------------------------------------------------------------
    # Research Question 1
    # -----------------------------------------------------------

    st.subheader("1️⃣ What is the average daily social media usage among students?")

    usage = df.groupby("Age")["Avg_Daily_Usage_Hours"].mean().reset_index()

    fig = px.area(
    usage,
    x="Age",
    y="Avg_Daily_Usage_Hours",
    template="plotly_dark",
    title="Average Daily Social Media Usage by Age"
    )

    fig.update_traces(
    line_color="Blue",
    fillcolor="rgba(0, 0, 255, 0.3)" 
    )

    st.plotly_chart(fig,width= "stretch")
    st.markdown("### 📖 Analysis")

    st.info("""
    1. The chart presents the average daily social media usage for each age group.

    2. It helps compare how usage varies among students of different ages.

    3. Peaks in the graph indicate age groups with higher average usage.

    4. The overall trend highlights differences in daily usage habits.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Social media usage differs across age groups.

    2. Some age groups spend more time online than others.

    3. Average usage provides a clearer comparison than individual observations.

    4. Age can be an important factor influencing social media behaviour.
    """)

    st.markdown("---")

    # ------------------------------------------------------------
    # Research Question 2
    # -----------------------------------------------------------

    st.subheader("2️⃣ Is there a relationship between daily social media usage and mental health score?")

    fig = px.scatter(
        df,
        x="Avg_Daily_Usage_Hours",
        y="Mental_Health_Score",
        title="Daily Usage vs Mental Health Score",
        labels={
            "Avg_Daily_Usage_Hours": "Average Daily Usage (Hours)",
            "Mental_Health_Score": "Mental Health Score"
        },
        opacity=0.8
    )

    fig.update_traces(
        marker=dict(
            color="Blue",
            size=10,
            line=dict(width=1, color='DarkSlateGrey')
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        xaxis_title="Avg_Daily_Usage_Hours",
        yaxis_title="Mental_Health_Score",
        font=dict(size=14)
    )

    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. Each point represents one student in the dataset.

    2. The x-axis shows average daily usage while the y-axis shows mental health score.

    3. Bubble size represents sleep duration.

    4. The chart helps identify possible relationships between social media usage and mental health.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Students show different mental health scores at different usage levels.

    2. No single usage level contains all observations.

    3. Sleep duration adds additional context to the relationship.

    4. Further statistical analysis can confirm the strength of the relationship.
    """)

    st.markdown("---")

    # ------------------------------------------------------------
    # Research Question 3
    # -----------------------------------------------------------

    st.subheader("3️⃣ How does social media usage affect sleep duration?")

    sleep = df.groupby("Age")["Sleep_Hours_Per_Night"].mean().reset_index()

    fig = px.line(
        sleep,
        x="Age",
        y="Sleep_Hours_Per_Night",
        markers=True,
        color_discrete_sequence=["royalblue"],
        title="Average Sleep Hours by Age"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Age",
        yaxis_title="Average Sleep Hours"
    )

    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The line chart displays the average sleep duration across different age groups.

    2. It helps identify age groups with higher or lower average sleep.

    3. Changes in the line indicate differences in sleeping habits.

    4. The chart provides an overview of sleep behaviour among students.
    """)

    st.markdown("### 💡 Insights")
    st.success("""
    1. Sleep duration varies between age groups.

    2. Some age groups receive more average sleep than others.

    3. Sleep is an important health indicator when studying social media usage.

    4. Comparing this chart with usage patterns can support further research on lifestyle habits.
    """)

    st.markdown("---")
    # -------------------------------------------------------------
    # Research Question 4
    # -------------------------------------------------------------

    st.subheader("4️⃣ Does excessive social media usage influence academic performance?")

    academic = (
            df.groupby("Affects_Academic_Performance")["Avg_Daily_Usage_Hours"]
            .mean()
            .reset_index()
        )


    fig = px.bar(
        academic,
        x="Affects_Academic_Performance",
        y="Avg_Daily_Usage_Hours",
        color="Affects_Academic_Performance",
        text="Avg_Daily_Usage_Hours",
        title="Average Daily Usage Based on Academic Performance",
        color_discrete_map={
            "No": "#1f77b4",   
            "Yes": "#4da6ff"   
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")

    fig.update_layout(
        xaxis_title="Academic Performance",
        yaxis_title="Average Daily Usage (Hours)",
        template="plotly_dark"
    )

    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The bar chart compares average daily social media usage across academic performance categories.

    2. It helps determine whether students reporting academic impact spend more time on social media.

    3. Differences in bar heights indicate variations in average usage.

    4. The chart provides an overview of the association between social media usage and academic performance.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Average daily usage differs across academic performance groups.

    2. Students reporting academic impact may have different usage patterns.

    3. The chart provides evidence for investigating academic outcomes.

    4. Additional factors may also influence academic performance.
    """)

    st.markdown("---")

    # -------------------------------------------------------------
    # Research Question 5
    # -------------------------------------------------------------

    st.subheader("5️⃣ Is there a significant relationship between sleep duration and mental health scores?")

    plot_df = df.copy()

    np.random.seed(42)

    plot_df["Sleep_Jitter"] = (
        plot_df["Sleep_Hours_Per_Night"]
        + np.random.uniform(-0.12, 0.12, len(plot_df))
    )

    plot_df["Mental_Jitter"] = (
        plot_df["Mental_Health_Score"]
        + np.random.uniform(-0.12, 0.12, len(plot_df))
    )

    fig = px.scatter(
        plot_df,
        x="Sleep_Jitter",
        y="Mental_Jitter",
        color="Gender",
        hover_data=["Sleep_Hours_Per_Night", "Mental_Health_Score"],
        title="Sleep Duration vs Mental Health Score",
        template="plotly_dark",
        color_discrete_sequence=["#1F77B4", "#4E79A7"]
    )

    fig.update_traces(
        marker=dict(
            size=7,
            opacity=0.6
        )
    )

    fig.update_layout(
        title_x=0.5,
        xaxis_title="Sleep Hours Per Night",
        yaxis_title="Mental Health Score"
    )

    st.plotly_chart(fig, width="stretch")

    

    st.markdown("### 📖 Analysis")

    st.info("""
    1. Each point represents an individual student.

    2. The x-axis shows sleep duration while the y-axis represents mental health score.

    3. Bubble size represents average daily social media usage.

    4. The scatter plot helps identify possible relationships between sleep and mental health.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Students with different sleep durations exhibit varying mental health scores.

    2. Sleep duration appears to influence overall well-being.

    3. Daily social media usage provides additional context for interpreting the relationship.

    4. Further statistical analysis can quantify the strength of this relationship.
    """)

    st.markdown("---")

    # -------------------------------------------------------------
    # Research Question 6
    # -------------------------------------------------------------

    st.subheader("6️⃣ Do male and female students differ in their social media usage patterns?")


    fig = px.box(
        df,
        x="Gender",
        y="Avg_Daily_Usage_Hours",
        color="Gender",
        points="all",
        title="Gender-wise Social Media Usage",
        color_discrete_map={
            "Male": "#1f77b4",
            "Female": "#4da6ff"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Gender",
        yaxis_title="Avg_Daily_Usage_Hours"
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The box plot compares daily social media usage between male and female students.

    2. The median line indicates the typical usage for each gender.

    3. The spread of each box shows variation in usage patterns.

    4. Outliers highlight students with exceptionally high or low usage.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Social media usage patterns vary between genders.

    2. The median usage helps compare the typical behavior of each group.

    3. Some students have unusually high or low usage regardless of gender.

    4. Gender-based comparisons provide useful demographic insights.
    """)

    st.markdown("---")
    # -------------------------------------------------------------
    # Research Question 7
    # -------------------------------------------------------------

    st.subheader("7️⃣ Which age group spends the most time on social media?")

    age_usage = df.groupby("Age")["Avg_Daily_Usage_Hours"].mean().reset_index()

    fig = px.bar(
        age_usage,
        x="Avg_Daily_Usage_Hours",
        y="Age",
        orientation="h",
        color="Avg_Daily_Usage_Hours",
       color_continuous_scale=px.colors.sequential.Blues,
        text_auto=".2f",
        title="Average Daily Social Media Usage by Age"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Average Daily Usage (Hours)",
        yaxis_title="Age"
    )

    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The horizontal bar chart compares average daily social media usage across different age groups.

    2. Longer bars indicate higher average daily usage.

    3. The visualization ranks age groups from lower to higher usage.

    4. It helps identify which age groups are more active on social media.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Social media usage is not the same across all age groups.

    2. Some age groups spend considerably more time online.

    3. The chart clearly highlights the highest-usage age group.

    4. Age plays an important role in understanding social media behaviour.
    """)

    st.markdown("---")

    # -------------------------------------------------------------
    # Research Question 8
    # -------------------------------------------------------------

    st.subheader("8️⃣ Which social media platform is most used?")

    platform = df["Most_Used_Platform"].value_counts().reset_index()
    platform.columns = ["Platform", "Users"]

    fig = px.pie(
        platform,
        names="Platform",
        values="Users",
        hole=0.5,
        color_discrete_sequence=px.colors.sequential.Blues,
        title="Most Used Social Media Platform"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The donut chart shows the proportion of users for each social media platform.

    2. Each slice represents one platform.

    3. Larger slices indicate platforms with more users.

    4. The chart provides an easy comparison of platform popularity.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. One or more platforms dominate student usage.

    2. Platform popularity varies across the dataset.

    3. Students use multiple social media platforms.

    4. Platform preference is useful for understanding user behaviour.
    """)

    st.markdown("---")

    # -------------------------------------------------------------
    # Research Question 9
    # -------------------------------------------------------------

    st.subheader("9️⃣ What is the overall impact of social media on students?")

    impact = df["Overall_Impact"].value_counts().reset_index()
    impact.columns = ["Impact", "Count"]

    fig = px.sunburst(
        impact,
        path=["Impact"],
        values="Count",
        color="Count",
        color_continuous_scale=px.colors.sequential.Blues,
        title="Overall Impact of Social Media"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The sunburst chart displays students' overall perception of social media.

    2. Each section represents an impact category.

    3. Larger sections indicate more responses.

    4. The visualization simplifies comparison among impact categories.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Students report different experiences with social media.

    2. The dominant impact category is immediately visible.

    3. Overall perception can be compared with mental health and sleep analyses.

    4. The chart summarizes students' opinions effectively.
    """)

    st.markdown("---")

    # -------------------------------------------------------------
    # Research Question 10
    # -------------------------------------------------------------

    st.subheader("🔟 Which country has the highest average daily social media usage?")

    country_usage = (
        df.groupby("Country")["Avg_Daily_Usage_Hours"]
        .mean()
        .reset_index()
    )

    
    country_usage["Country"] = country_usage["Country"].replace({
        "USA": "United States",
        "UK": "United Kingdom",
        "UAE": "United Arab Emirates",
        "South Korea": "Korea, South",
        "Russia": "Russian Federation"
    })

    fig = px.choropleth(
        country_usage,
        locations="Country",
        locationmode="country names",
        color="Avg_Daily_Usage_Hours",
        hover_name="Country",
        hover_data={
            "Avg_Daily_Usage_Hours":":.2f"
        },
        color_continuous_scale=px.colors.sequential.Blues,
        title="Average Daily Social Media Usage by Country"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="white",
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)"
        ),
        coloraxis_colorbar=dict(
            title="Avg Usage<br>(Hours)"
        )
)


    st.plotly_chart(fig,width= "stretch")

    st.markdown("### 📖 Analysis")

    st.info("""
    1. The chart compares average daily social media usage among countries.

    2. Countries are ranked based on average usage.

    3. Longer bars indicate higher average daily usage.

    4. It highlights geographical differences in social media behaviour.
    """)

    st.markdown("### 💡 Insights")

    st.success("""
    1. Social media usage differs between countries.

    2. Some countries have noticeably higher average usage.

    3. Country-level comparisons provide valuable global insights.

    4. Cultural and lifestyle differences may contribute to usage variations.
    """)

    st.markdown("---")

# -------------------------------------------------------------
# Prediction
# -------------------------------------------------------------


elif menu == "🔮 Prediction":
    

    st.title("🔮 Mental Health Score Prediction")
    st.write(
        "Predict a person's mental health score using "
        "social media usage and lifestyle details."
    )

    # Load dataset
    df = pd.read_csv("social_media_impact_on_health_cleaned.csv")

    target = "Mental_Health_Score"

    if target not in df.columns:
        st.error(f"Target column '{target}' was not found.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # Remove rows without a target value
    df = df.dropna(subset=[target])

    # Select useful input features
    possible_features = [
        "Age",
        "Gender",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Affects_Academic_Performance",
        "Overall_Impact",
        "Most_Used_Platform",
        "Academic_Level",
        "Country",
    ]

    features = [
        col for col in possible_features
        if col in df.columns and df[col].notna().any()
    ]

    if len(df) < 10 or not features:
        st.error("Not enough usable data to train the prediction model.")
        st.stop()

    X = df[features].copy()
    y = df[target]

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        col for col in features
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("num", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("cat", categorical_pipeline, categorical_features)
        )

    preprocessing = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Train and evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    st.subheader("📊 Model Performance")
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")

    st.caption(
        "MAE shows the average prediction error in the target "
        "score's units. This is an estimate, not a medical diagnosis."
    )

    # User input form
    st.subheader("🧑 Enter Your Details")

    with st.form("prediction_form"):
        user_data = {}

        for col in features:
            if col in numeric_features:
                valid_values = pd.to_numeric(
                    df[col], errors="coerce"
                ).dropna()

                if valid_values.empty:
                    user_data[col] = 0
                else:
                    minimum = float(valid_values.min())
                    maximum = float(valid_values.max())
                    default = float(valid_values.median())

                    if minimum == maximum:
                        user_data[col] = st.number_input(
                            col, value=default
                        )
                    else:
                        user_data[col] = st.number_input(
                            col,
                            min_value=minimum,
                            max_value=maximum,
                            value=min(max(default, minimum), maximum)
                        )
            else:
                choices = df[col].dropna().astype(str).unique().tolist()

                if choices:
                    user_data[col] = st.selectbox(col, choices)
                else:
                    user_data[col] = ""

        submitted = st.form_submit_button(
            "Predict Mental Health Score"
        )

    if submitted:
        input_df = pd.DataFrame([user_data], columns=features)
        result = float(model.predict(input_df)[0])

        st.success(f"Predicted Mental Health Score: {result:.2f}")

        st.info(
            "This prediction is based on patterns in the project "
            "dataset. It is for educational purposes only."
        )


elif menu == "📌 Research Questions Summary":
    st.subheader("📌 Research Questions Summary")

    st.success("""
    ### Key Findings

    ✅ Average daily social media usage varies across different age groups.

    ✅ Social media usage, sleep duration, and mental health show meaningful patterns that can be explored further.

    ✅ Academic performance differs across students with different usage levels.

    ✅ Male and female students display different social media usage patterns.

    ✅ Platform preference varies, with some platforms being considerably more popular.

    ✅ Students have mixed opinions regarding the overall impact of social media.

    ✅ Average daily usage also varies across countries.

    ✅ The dashboard provides useful evidence for understanding the relationship between social media usage and student health.
    """)

# --------------------------------------------------------------
# 📌 CONCLUSION
# --------------------------------------------------------------
elif menu == "📌 Conclusion":
    st.title("📌 Conclusion")
    st.markdown("---")

    st.header("📖 Project Summary")

    st.write("""
    This project analyzed the impact of social media usage on students'
    health and academic life using Exploratory Data Analysis (EDA) and
    interactive Plotly visualizations.

    The dashboard explored relationships among daily social media usage,
    mental health, sleep duration, academic performance, age, gender,
    country, and preferred social media platform.

    The analysis provides valuable insights into students' social media
    behaviour and highlights important trends that may help educators,
    students, and researchers better understand digital lifestyle patterns.
    """)

    st.markdown("---")

    # -------------------------------------------------------------
    # Key Findings
    # -------------------------------------------------------------

    st.header("🔍 Key Findings")

    st.success("""
    ### 1️⃣ Daily Social Media Usage
    Most students spend a moderate amount of time on social media every day,
    while a smaller group spends significantly longer hours online.
    """)

    st.success("""
    ### 2️⃣ Mental Health
    Mental health scores vary among students, suggesting differences in
    well-being that may be associated with lifestyle and social media habits.
    """)

    st.success("""
    ### 3️⃣ Sleep Duration
    Students who spend more time on social media often show noticeable
    differences in sleeping patterns.
    """)

    st.success("""
    ### 4️⃣ Academic Performance
    Many students believe that excessive social media usage influences their
    academic performance.
    """)

    st.success("""
    ### 5️⃣ Platform Preference
    Students use different social media platforms, with a few platforms
    being considerably more popular than others.
    """)

    st.markdown("---")

# -------------------------------------------------------------
# Recommendations
# -------------------------------------------------------------

    st.header("💡 Recommendations")

    st.info("""
    ### 1.Students should maintain a balanced daily screen time.

    ### 2.Educational institutions should promote digital wellness programs.

    ### 3.Students should develop healthy sleeping habits.

    ### 4.Parents and teachers should encourage responsible social media use.

    ### 5.Mental health awareness activities should be integrated into schools and colleges.
         """)

    st.markdown("---")

# -------------------------------------------------------------
# Future Scope
# -------------------------------------------------------------

    st.header("🚀 Future Scope")

    st.warning("""
    ### Future Improvements

    • Collect larger datasets from different countries.

    • Include additional health indicators such as stress and anxiety.

    • Apply Machine Learning models to predict mental health outcomes.

    • Develop a real-time interactive dashboard.

    • Perform longitudinal studies to observe changes over time.
    """)

    st.markdown("---")

# -------------------------------------------------------------
# References
# -------------------------------------------------------------

    st.header("📚 References")

    st.write("""
    1. Pandas Documentation

    https://pandas.pydata.org/

    2. Plotly Documentation

    https://plotly.com/python/

    3. Streamlit Documentation

    https://docs.streamlit.io/

    4. Python Documentation

    https://docs.python.org/3/

    5. Kaggle Dataset Repository

    https://www.kaggle.com/
    """)

    st.markdown("---")

# -------------------------------------------------------------
# Final Conclusion
# -------------------------------------------------------------

    st.header("🎯 Final Conclusion")

    st.success("""
    The analysis demonstrates that social media usage is closely associated
    with several aspects of students' lives, including sleep habits, mental
    health, and academic performance. While social media provides valuable
    opportunities for communication and learning, excessive usage may have
    undesirable effects. Maintaining a healthy balance between online
    activities and daily responsibilities can contribute to improved
    well-being and academic success.
    """)

    st.markdown("---")

# -------------------------------------------------------------
# Thank You
# -------------------------------------------------------------

    st.balloons()

    st.markdown(
        "<h2 style='text-align:center;color:green;'>🎉 Thank You for Visiting the Dashboard 🎉</h2>",
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div style="text-align:center; font-size:15px; color:#6c757d;">
        <b>Developed using Python, Pandas, Plotly and Streamlit.</b>
    </div>
    """,
    unsafe_allow_html=True
)
