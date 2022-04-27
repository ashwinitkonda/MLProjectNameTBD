# ML Approaches to Predict the Occurrence of Type 2 Diabetes
Introduction/Background:
In this report, we are predicting the occurrence of diabetes based on lifestyle which involves macronutrient breakdown and exercise activity. There is prior work done regarding diabetes classification and predictive analysis for diabetes. Classification of diabetes is a crucial step for predicting diabetes, however it’s a difficult task since it involves analyzing multiple different factors which can lead to inaccurate results. Several techniques were used for diabetes prediction including a stacking-based ensemble method as well as data mining techniques such as random forest, logistic regression, and naïve Bayes. These techniques outperformed the state-of-the-art approach with 79% accuracy. The dataset that we plan to use is the NHANES 2017-March 2020 Pre-Pandemic Questionnaire Data which includes many features such as blood sugar levels, age, insulin, diet, and physical activity.
 
Problem Definition:
An energy-conscious diet combined with modest physical activity improves various metabolic syndrome indices and delays the onset of diabetic complications. Over the last few decades, the prevalence of illnesses linked to insulin resistance, such as the metabolic syndrome and type 2 diabetes, has risen. In college, it is very easy to be overwhelmed with school work, extracurricular activities and other priorities and so our focus on our health takes a back seat. College students experience an array of mental and physical health issues due to various factors including overwhelming stress to succeed in classes as well as personal life changes. Having to manage all these different situations can be extremely intimidating for most students. Therefore, they start developing unhealthy lifestyle habits including eating more fast food, exercising less and increased alcohol consumption. Therefore, we wanted to complete this project to allow students to predict whether their current lifestyle is sustainable or if it will result in them developing diabetes which would hinder their lives forever. 

(Mid-Point Check-In):

Dataset Modification Rationale: 
While our project proposal discussed the macro-scale dataset, we found that the National Health and Nutrition Examination Survey (NHANES) dataset lacked statistical completeness for the laboratory medical diagnostic data points, which are key to a complete picture of a diabetes diagnosis (many patients were only surveyed for these medical biomarkers if they were already diagnosed with diabetes). For this reason, we chose to focus on the Pima Indians Diabetes Database for our initial model development, training, and evaluation efforts, and we have documented this progress in our midpoint report. That said, we have continued our exploration of the NHANES dataset in parallel and plan to compare these findings and explore the potential interplay between biomarkers and lifestyle factors in our next project phase. 
 
Data Collection and Cleaning
In order to acquire and clean the data, we started by compiling the Pima Indian Diabetes dataset (.csv files) to use within our data cleaning notebook. Next, we compiled the files into a sorted dictionary of data frames in order to conduct recategorization and row dropping transformations at scale. We identified key variables within each year of data (the exact feature labels varied between years, so this reclassification would prove important for consistency across the entire dataset, and for further feature reduction and model operations). 
We wrote a few key cleaning functions - ‘recategorize,’ ‘count_ vals,’ and ‘drop_rows,’ which we applied across the sorted dictionary. We then created a copy of the dataframes, and dropped columns that were unrelated to our analytic objective (connecting physical activity and macronutrient composition to the occurrence of diabetes). Next, we renamed the columns (feature labels) across all of the dataframes, then added the years as a column to each of the data frames. We then concatenated the frames, creating a final data frame containing the data across all of the years of available data. 
We then removed correlated features, and any rows with missing values using the aforementioned ‘drop_rows’ function, applied to a new cumulative data frame called ‘result_cleaned.’ To view the percentage of removed values, we simply calculated the length of both data frames (number of rows) before and after cleaning, then subtracted and normalized this value to derive the efficacy of our cleaning and the completeness of our final dataset. 
 
Methods:
The first step will be gathering data about what factors are important in diagnosing diabetes. Then, after researching which factors are important in the diagnoses, we will try a few methods of supervised learning to see what results in the most accuracy for our dataset. Given that our data is classification based, since a patient can be diagnosed with diabetic, prediabetic, and not diabetic, we will try models such as random forest and logistic regression and determine which results in more accuracy for our project. Both models have their advantages and disadvantages and trying them out will determine which blends better with our dataset.

(Mid-Point Check-In):

Final Preprocessing:
The first phase in the process is data preprocessing, which marks the start of the procedure. Data preprocessing helps to clean, format, and organize the raw data by removing all the inconsistencies, inaccuracies, and missing data thereby making it ready for Machine Learning models. We acquired the Pima Indians Diabetes Database dataset during our initial research and then imported the relevant libraries. The libraries we imported that proved to be crucial were NumPy, Pandas, and Matplotlib. The next step was to detect and handle missing values effectively to avoid drawing incorrect conclusions and inferences from the data. This was accomplished by deleting a specific row with a null value for a feature or a specific column with more than 75% of the data missing, within our drop_row function. Splitting the dataset was the next step we completed. We split the data into a 80:20 ratio with taking 80% of the data for training the model and the balance 20% left for testing. Feature scaling was the final step in the pre-processing stage of this project. It was important to feature scale to limit the range of variables so that our group could compare them on the same level. We used the normalization method to do this. We used the sci-kit-learn library to help us do this. All the variables in the output are scaled between the values 0 and 1.

PCA:
We used PCA to reduce the dimensionality of our dataset from 8 categories (not including the outcome data) and then condensing it to 7 categories of data. We then combined this data with the outcome column to create our final dataset. The reason we chose to do this is because by reducing the dataset with this method, we are coupling data that goes hand-in-hand in order to identify relationships in the data and helps maintain a standard throughout the dataset.

<img width="323" alt="image" src="https://user-images.githubusercontent.com/100390257/165456721-e7129dec-18c0-4976-aeb1-a5ba6bba6c08.png">

 
Logistic Regression:
Logistic Regression is used to analyze the data presented and return a probability value of how likely it is to get a certain outcome. In this case we are looking at the data and figuring out the probability of predicting diabetes. How we used this method was by fitting the training data of X to the training data of y and then predicting the class labels of X.
 
Random Forest:
Random Forest was another method we tried, it was implemented similarly to Logistic Regression, but essentially what it does is turn the data into decision trees and makes its predictions based off of that. Because of this implementation Random Forest would probably work better on categorical data, which we do not have, so this may not be the most efficient way to predict diabetes. Aside from that, the more trees we create with this method the higher the accuracy should be so we tried it with 5, 10, 50, and 100 n_estimators in order to compare the results.

SVC:
The last method we implemented was the linear SVC method. This takes the data provided and finds a linear ‘best fit’ line that simplifies the data into one set of dependent and one set of independent data. This way the data is more easily analyzed as well as visualized. We implement this the same way as the last few methods as well by fitting the data and predicting the labels to find our accuracy.

Potential Results/Discussion:

The potential results for this project is to build a program that will be able to take into account certain lifestyle markers of an individual such as diet, physical exercise, alcohol consumption, etc. Using these factors we hope to predict if an individual is at risk of getting diagnosed with diabetes and if so, what specific type(s) of diabetes they are at risk for. What we need to focus on is the types of markers we need to collect in order to make these observations as well as solidifying which datasets to use in order to train our models. With more research we should be able to narrow down exactly what we hope for our program to calculate and predict.

(Mid-Point Check-In)
To obtain our results, we ran three different machine learning models: logistic regression, random forest, and support vector classification. They resulted in similar accuracies ranging between 70% to 80% with different ones doing slightly better than the other two each time the models were run. The random forest model, we varied the number of trees with four options: 5,10,50,100. They all performed similarly well on our dataset except 50 trees estimators performed slightly better than the other options. The graph of our results can be seen here.

 
References:
 
Umair Muneer Butt, Sukumar Letchmunan, Mubashir Ali, Fadratul Hafinaz Hassan, Anees
Baqir, Hafiz Husnain Raza Sherazi, "Machine Learning Based Diabetes Classification and Prediction for Healthcare Applications", Journal of Healthcare Engineering, vol. 2021, Article ID 9930985, 17 pages, 2021. https://doi.org/10.1155/2021/9930985
 
World Bank. "Health Nutrition And Population Statistics" World Development Indicators. The
World Bank Group, 23 Feb, 2022, datacatalog.worldbank.org/search/dataset/0037652. Accessed 23 Feb. 2022.
 
Michael Kahn. Diabetes Data Set. UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml/datasets/diabetes]. Irvine, CA: University of California, School of Information and Computer Science.
 
Ahmadi, F., Ganjkhanloo, F., &amp; Ghobadi, K. (n.d.). (rep.). An Open-Source Dataset on
Dietary Behaviors and DASH Eating Plan Optimization Constraints. https://arxiv.org/pdf/2010.07531.pdf
 
Timeline:
 
MODEL 1 DESIGN & SELECTION:

Data Cleaning
- Rhea
- 8 Days
Data Visualization
- Sahya
- 10 Days
Feature Reduction
- Josh
- 10 Days
Coding & Implementation
- Swetha & Ashwini
- 20 Days
Results Evaluation
- Rhea & Josh
- 1-2 Days


MODEL 2 DESIGN & SELECTION:

Data Cleaning
- Swetha
- 8 Days
Data Visualization
- Josh
- 10 Days
Feature Reduction
- Ashwini
- 10 Days
Coding & Implementation
- Rhea & Sahya
- 20 Days
Results Evaluation
- Swetha & Ashwini
- 1-2 Days



MODEL 3 DESIGN & SELECTION:

Data Cleaning
- Rhea
- 8 Days
Data Visualization
- Sahya
- 10 Days
Feature Reduction
- Swetha
- 10 Days
Coding & Implementation
- Ashwini & Josh 
- 20 Days
Results Evaluation & Model Comparison
- Sahya & Josh
- 2-3 Days


