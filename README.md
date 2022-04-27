# ML Approaches to Predict the Occurrence of Type 2 Diabetes: Final Report
Ashwini Thirukkonda, Sahya Nara, Josh Patel, Rhea Ganguli, Swetha Mohandas

## Introduction/Background 

In this report, we have conducted an in-depth analysis of the most optimal type 2 diabetes (T2D) dietary and lifestyle interventions, correlated to diet (dietary macronutrient composition), activity, and general health risk factors. Prior work in the diabetes classification and predictive analysis domain is built on genetic data in laboratory settings and narrowly sampled physical health, environmental, and behavioral determinant modeling. To this, the genetic and non-genetic determinants of T2D have not been fully elucidated with machine learning models, especially when leveraging large health data repositories (‘big data’). 

We structured our initial analysis around the tightly-controlled, bio-marker based Pima Indians Diabetes Dataset, using the following techniques: PCA for dimensionality-reduction, logistic regression, random forest for supervised ensemble learning, and support vector classification. After evaluating the performance of these models, we set out to apply select models to the NHANES (National Health and Nutrition Examination Survey) longitudinal dataset, which integrates 19 collections of data, and 67 features, including many lifestyle variables, such as dietary macronutrient breakdowns, physical activity metrics, and alcohol consumption, as well as certain key biomarkers, including diabetes diagnoses, cholesterol levels, and fasting insulin levels. After rigorous data cleaning and preprocessing, we implemented and evaluated the performance of two classification models: Random Forest and XGBoost using the respective F-1 scores, and compiled a combined weighted analysis using the results of our chi-squared feature analysis, Random Forest model, and XGBoost model. 

## Problem definition

Globally, T2D is a major public health concern, with a host of complications and comorbidities impacting over 537,000,000 million adults worldwide - which scales to 1 in 10 adults globally. This figure is predicted to rise to 783,000,000 million adults by 2045, with 3 in 4 diagnosed individuals residing in low to middle-income nations. Therefore, it is critically important to develop high-performing ML-based predictive models of T2D with clear insights to apply in clinical and community interventions. 
Extensive research has shown that an energy-conscious diet combined with modest physical activity improves various metabolic syndrome indices (including insulin resistance) and delays the onset of diabetic complications. By implementing timely and effective diagnosis, the health outcomes of millions can be improved, hospitalizations can be prevented, and the quality of life for a substantial subset of humanity can be bolstered. To this, classification-based machine learning models can aid in improved T2D prognostication by expanding the diagnostic scope beyond biomarkers and laboratory data, and integrating lifestyle factors. 
While T2D is more prevalent in middle-late aged adults, the formation of lifestyle habits leading to insulin resistance and T2D begins in college. As students, it is easy to be overwhelmed with school work, extracurricular activities and other priorities, leading to increased levels of stress and mental/physical health concerns. This leads to the formation of unhealthy lifestyle habits, including adopting a high-glycemic and high-caloric diet, exercising less, and increasing alcohol consumption. While the college-aged population is a specialized case, the findings of our population-scale analysis can shed light on the applicability of T2D predictive analysis to early (and potentially life-altering) health interventions. 

## Overview of Datasets

**Granular population-specific dataset: Pima Indians Diabetes Database**

This dataset was used for our initial analysis and model development. Originally from the National Institute of Diabetes and Digestive and Kidney Diseases, features include key medical diagnostic measurements (number of pregnancies the patient has had, BMI, insulin level, age, etc). Moreover, all Pima Indian population patient data points are gathered from females who are at least 21 years old.
 
**Macro-Scale dataset: National Health and Nutrition Examination Survey (NHANES)**

This dataset was used to apply our models to a more expansive feature set and population, with a longer time-scale and wider scope. Continuous NHANES initiated in 1999, with data available until 2018 - ongoing with a sample of 5000 participants annually. Sampling utilizes a nationally representative civilian sample identified through multistage probability sampling design. The data is hosted on the NHANES webpages as raw data files (.xpt), and the key data collection categories are: Examination, Laboratory, Questionnaire, Demographics, and Dietary. Each data collection has a codebook of features (abbreviated and mapped with an alphanumeric code). 

## Granular Population-Specific Dataset: Pima Indians Diabetes Database

### **Data Collection and Cleaning**
Our PIDD was acquired from UCI Machine Learning’s repository, housed on Kaggle. The first phase is data preprocessing, which marks the start of the procedure. Data preprocessing helps to clean, format, and organize the raw data by removing all the inconsistencies, inaccuracies, and missing data thereby making it ready for Machine Learning models. We acquired the Pima Indians Diabetes Database dataset during our initial research and then imported the relevant libraries. The libraries we imported that proved to be crucial were NumPy, Pandas, and Matplotlib. 
The next step was to detect and handle missing values effectively to avoid drawing incorrect conclusions and inferences from the data. This was accomplished by deleting a specific row with a null value for a feature or a specific column with more than 75% of the data missing. Splitting the dataset was the next step we completed. We split the data into a 80:20 ratio with taking 80% of the data for training the model and the balance 20% left for testing.
Feature scaling was the final step in the pre-processing stage of this project. It was important to feature scale to limit the range of variables so that our group could compare them on the same level. We used the normalization method to do this. We used the sci-kit-learn library to help us do this. All the variables in the output are scaled between the values 0 and 1.

### **Methods**
**PCA**

We used PCA to reduce the dimensionality of our dataset from 8 categories (not including the outcome data) and then condensing it to 7 categories of data. We then combined this data with the outcome column to create our final dataset. The reason we chose to do this is because by reducing the dataset with this method, we are coupling data that goes hand-in-hand in order to identify relationships in the data and helps maintain a standard throughout the dataset.

<img width="323" alt="image" src="https://user-images.githubusercontent.com/100390257/165456721-e7129dec-18c0-4976-aeb1-a5ba6bba6c08.png">

**Logistic Regression:**

Logistic Regression is used to analyze the data presented and return a probability value of how likely it is to get a certain outcome. In this case we are looking at the data and figuring out the probability of predicting diabetes. How we used this method was by fitting the training data of X to the training data of y and then predicting the class labels of X.

<img width="367" alt="image" src="https://user-images.githubusercontent.com/100390257/165457215-1cfeb389-c578-40f7-8283-ec25cb2d9b61.png">
 
**Random Forest:**

Random Forest was another method we tried, it was implemented similarly to Logistic Regression, but essentially what it does is turn the data into decision trees and makes its predictions based off of that. Because of this implementation Random Forest would probably work better on categorical data, which we do not have, so this may not be the most efficient way to predict diabetes. Aside from that, the more trees we create with this method the higher the accuracy should be so we tried it with 5, 10, 50, and 100 n_estimators in order to compare the results.

<img width="562" alt="image" src="https://user-images.githubusercontent.com/100390257/165457359-c34a0e87-17fa-43b3-a49b-b51a3bc7f1fa.png">

**SVC:**

The last method we implemented was the linear SVC method. This takes the data provided and finds a linear ‘best fit’ line that simplifies the data into one set of dependent and one set of independent data. This way the data is more easily analyzed as well as visualized. We implement this the same way as the last few methods as well by fitting the data and predicting the labels to find our accuracy.

<img width="320" alt="image" src="https://user-images.githubusercontent.com/100390257/165457480-51307367-926c-4b56-8519-41e05092531f.png">

**Results & Discussion:**

To obtain our results, we ran three different machine learning models: logistic regression, random forest, and support vector classification. They resulted in similar accuracies ranging between 70% to 80% with different ones doing slightly better than the other two each time the models were run. The random forest model, we varied the number of trees with four options: 5,10,50,100. They all performed similarly well on our dataset except 50 trees estimators performed slightly better than the other options. The graph of our results can be seen here.

<img width="474" alt="image" src="https://user-images.githubusercontent.com/100390257/165457622-82178efd-17f3-4cdb-9d45-95f8b699c260.png">

The accuracy of the other models, logistic regression and SVC,  can be seen here:

<img width="567" alt="image" src="https://user-images.githubusercontent.com/100390257/165457761-e56564c6-b7c6-4f15-a68d-8d7892b4af28.png">

From running our models, we were able to determine that the two most significant factors for a diabetes diagnosis are average number of pregnancies and average blood insulin levels. It’s important to note that these results are found from analysis of the Pima Indian Diabetes Dataset, and are representative of this female population.

<img width="540" alt="image" src="https://user-images.githubusercontent.com/100390257/165457940-b8476ab6-c08b-4792-8969-77f866e9e7f7.png">

## Macro-Scale Dataset: National Health and Nutrition Examination Survey (NHANES)

To further our analysis and increase the applicability of our findings, we continued our modeling work with the NHANES dataset, following the same cleaning, preprocessing, splitting, feature scaling, and model implementation and evaluation approach. Our methods and conclusions are detailed in the following sections. 

### Data Collection and Cleaning

The first stage was data acquisition, in which we used a Python script to scrape the NHANES website, parse each webpage (1999-2018, two year increments) for links to .XPT files, download these files, and convert them to .CSV files. In our cleaning notebooks, we mapped the column labels of each of the 19 data collections to feature descriptions, renamed the features and standardized them across the years. Following this, we removed all ‘NaN,’ ‘Refused,’ ‘Don’t Know’ data points and quantified the removal to less than 10% of the combined data set; a sample cleaned data frame from our physical diagnostic data collection is shown below. In order to address the categorical data, we conducted a one-hot encoding cleaning exercise, in which we assigned 1s and 0s to categorical values in order to provide the ML algorithm with numerical data. We then uploaded the cleaned data collections to a local NoSQL (MongoDB) database, which housed over 14,000 rows, 19 collections, and 67 features from the processed NHANES dataset.

[IMAGE - Cleaned Data Frame]

### Preliminary Data Analysis

In order to evaluate the efficacy of our cleaning and preprocessing, and to gain insight into the quality of the dataset (variance, completeness) across key dimensions, we conducted the following preliminary analyses. Primarily, we evaluated the class distribution of diabetes diagnoses (DIQ010) cumulatively and on a year-over-year basis (figures 1 and 2), followed by a box-plot analysis of the numerical features (figure 3) and a bar chart analysis of the categorical features (figure 4). Through this analysis, we confirmed that our dataset was sufficiently unbiased and complete for our predictive modeling. 

[IMAGE - Preliminary Plots]

### Extended Preprocessing: Up/Down Sampling

To prepare our dataset for model implementation, we conducted up and downsampling using the sci-kit learn based imbalanced learn library. The rationale for this was as follows: since we are working with health data, the classes are imbalanced (more observations for certain key classes/data collections), and up and downsampling techniques can be used to augment and better fit the data. Our upsampling technique was SMOTE, or Synthetic Minority Over-sampling Technique, which considers the current point and its k-nearest neighbors to create a synthetic data point using features between them. Our downsampling technique was Tomek links, which works by removing overlap between classes, ensuring that all majority class links are removed until all minimally distanced nearest pairs are of the same class. This allows for a border between the classes, creating less data overall.

For feature selection and dimensionality reduction, we conducted chi-squared analyses using the ‘SelectKBest’ class; our implementation is shown on the right. From this analysis, we gained one perspective on the ranked feature importance across the NHANES dataset. In order of importance, the chi-squared analysis found the top risk factors of T2D to be dietary caloric (and other nutrient) consumption markers, age, and alcohol consumption. This analysis was used to further distill our feature set, and the scores were integrated into our weighted risk factor analysis (following the classification model implementation). 

[IMAGE - Chi Squared Analysis]

### Random Forest Model: Classification

The first model we implemented was Random Forest, a commonly used supervised learning and ensemble method for classification. ‘Random Forests’ are collections of decision trees, each containing randomly selected subsets of features. This model’s output is the majority vote of the classes, and each decision tree is trained with bagging - a process by which the forest selects a random sample of 2/3 of the training data (with replacement), and the remaining ⅓ of the data (out-of-bag) is used to produce generalization error. The benefit of bagging is clear - it decreases the variance of model without increasing variance of the dataset. Furthermore, the features are split via a metric called Gini importance (the total reduction of criteria brought about by the feature). To handle missing values and indicate the feature importance, we used the scikit-learn library. 

We implemented our model on the regular, upsampled, and downsampled data (implementation shown below), and used the GridSearch CV class for hyperparameter tuning. After finding these best parameters, we fit the Random Forest Classifiers to the data, the output being the top-10 feature importance plot (shown below). 

[IMAGE - Feature Importance Plot]
[IMAGE - Downsampling Implementation]

### XGBoost Model: Classification

After this, we implemented the XGBoost model, a commonly used classification technique. Gradient Boosting uses weighting to select trees that are ‘weak learners’ and converts them into ‘strong learners,’ which reduces errors in the previous trees. Moreover, by using boosting, we can prevent overfitting with trees that have fewer splits, improving model performance and execution speed. 

We used the GridSearch CV class for hyperparameter tuning to find the best hyperparameters from the regular, upsampled, and downsampled data. After finding these best parameters, we fit the XGBoost model to the NHANES data, the output being our top-10 feature importance plot (shown on the right). 

[IMAGE - XGBoost Implementation]
[IMAGE - Feature Importance Plot]

### Model Evaluation and Comparison

In order to evaluate the models, we used the F-1 scores generated from each. The F-1 score is the harmonic mean of precision and recall, calculated with the following formula: 2TP/(2TP+FP+FN). The key evaluative metrics of each model are shown below: 

[IMAGE - Evaluation Table]

Between the two models, the Random Forest classifier model marginally outperformed the XGBoost model, with an improved precision and accuracy. 

## Diabetes Risk Factors: Results and Discussion

In order to extract definitive insights from our model fitting exercises, we conducted a combined analysis of the top ten risk factors for T2D. To accomplish this, we used a weighted formula from the respective F-1 scores: 0.025 [Chi-2] + .15 [Reg RF + Up RF + Down RF] + .10[Reg XGB F1 + Up XGB F1 + Down XGB F1]  + 0.075 [Reg XGB FI + Up XGB FI + Down XGB FI]. This resulted in the following feature importance plot (shown on the right), which has informed our analytical conclusions. To clarify the feature abbreviations, the top risk factors were shown to be: Age, High Blood Pressure (Categorical), Times Received Healthcare In Last Year, General Health Condition, BMI, HDL (Circulating Cholesterol), Dietary Carbohydrate Intake, Dietary Fiber Intake, Vitamin C Intake, and Systolic Blood Pressure. 

As shown by the previous model outputs and the combined risk factor analysis, macronutrient-based nutrition habits (high carbohydrate intake, low fiber intake) are some of the most important risk factors associated with the diagnosis of diabetes, along with key biomarkers (age, blood pressure, BMI, cholesterol levels). These findings add a layer of complexity to our T2D treatment, prognostication, and diagnosis methodologies, highlighting that long-term lifestyle habits have a clear association with the development of this debilitating disease. Moreover, this analysis indicates that further exploration of dietary-specific datasets is merited and pertinent to the prognostication of T2D, which is something we’re deeply curious about (and plan on pursuing as future work). 

[IMAGE - Diabetes Risk Factors]

There are many ways to expand upon this project, such as considering a wider array of features and risk factors, extrapolating our findings to other population samples, and exploring alternate models and methods for improved performance. As shown in this report, by synergizing these insights with optimized modelling techniques and more granular datasets, it is possible to optimize population health management for Type 2 Diabetes (and other widespread conditions).

## References:
 
Umair Muneer Butt, Sukumar Letchmunan, Mubashir Ali, Fadratul Hafinaz Hassan, Anees
Baqir, Hafiz Husnain Raza Sherazi, "Machine Learning Based Diabetes Classification and Prediction for Healthcare Applications", Journal of Healthcare Engineering, vol. 2021, Article ID 9930985, 17 pages, 2021. https://doi.org/10.1155/2021/9930985
 
World Bank. "Health Nutrition And Population Statistics" World Development Indicators. The
World Bank Group, 23 Feb, 2022, datacatalog.worldbank.org/search/dataset/0037652. Accessed 23 Feb. 2022.
 
Michael Kahn. Diabetes Data Set. UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml/datasets/diabetes]. Irvine, CA: University of California, School of Information and Computer Science.
 
Ahmadi, F., Ganjkhanloo, F., &amp; Ghobadi, K. (n.d.). (rep.). An Open-Source Dataset on
Dietary Behaviors and DASH Eating Plan Optimization Constraints. https://arxiv.org/pdf/2010.07531.pdf
 
## Timeline:
 
<img width="577" alt="image" src="https://user-images.githubusercontent.com/100390257/165458948-7bde5249-46b5-46a8-9a34-cc817550e2da.png">
<img width="583" alt="image" src="https://user-images.githubusercontent.com/100390257/165458992-0aa3ef77-0d50-409d-8703-d6113886e8dc.png">
<img width="575" alt="image" src="https://user-images.githubusercontent.com/100390257/165459005-a2d196cc-bea3-4850-9b06-7473ae6bca2b.png">

