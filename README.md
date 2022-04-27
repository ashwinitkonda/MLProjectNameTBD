# ML Approaches to Predict the Occurrence of Type 2 Diabetes
Introduction/Background 
In this report, we are predicting the occurrence of diabetes based on lifestyle which involves macronutrient breakdown and exercise activity. There is prior work done regarding diabetes classification and predictive analysis for diabetes. Classification of diabetes is a crucial step for predicting diabetes, however it’s a difficult task since it involves analyzing multiple different factors which can lead to inaccurate results. Several techniques were used for diabetes prediction including a stacking-based ensemble method as well as data mining techniques such as random forest, logistic regression, and support vector classification. These techniques outperformed the state-of-the-art approach with 79% accuracy. The dataset that we plan to use is the NHANES 2013-March 2014 Questionnaire Data and the Pima Indians Diabetes Dataset which includes many features such as blood sugar levels, age, insulin, diet, and physical activity.
 
Problem definition
An energy-conscious diet combined with modest physical activity improves various metabolic syndrome indices and delays the onset of diabetic complications. Over the last few decades, the prevalence of illnesses linked to insulin resistance, such as the metabolic syndrome and type 2 diabetes, has risen. In college, it is very easy to be overwhelmed with school work, extracurricular activities and other priorities and so our focus on our health takes a back seat. College students experience an array of mental and physical health issues due to various factors including overwhelming stress to succeed in classes as well as personal life changes. Having to manage all these different situations can be extremely intimidating for most students. Therefore, they start developing unhealthy lifestyle habits including eating more fast food, exercising less and increased alcohol consumption. Therefore, we wanted to complete this project to allow students to predict whether their current lifestyle is sustainable or if it will result in them developing diabetes which would hinder their lives forever. 

Dataset Modification Rationale
While our project proposal discussed the macro-scale dataset, we found that the National Health and Nutrition Examination Survey (NHANES) dataset lacked statistical completeness for the laboratory medical diagnostic data points, which are key to a complete picture of a diabetes diagnosis (many patients were only surveyed for these medical biomarkers if they were already diagnosed with diabetes). For this reason, we chose to focus on the Pima Indians Diabetes Database for our initial model development, training, and evaluation efforts, and we have documented this progress in our midpoint report. That said, we have continued our exploration of the NHANES dataset in parallel and plan to compare these findings and explore the potential interplay between biomarkers and lifestyle factors in our next project phase.
 
Data Collection and Cleaning
In order to acquire and clean the data, we started by compiling the Pima Indian Diabetes dataset (.csv files) to use within our data cleaning notebook. Next, we compiled the files into a sorted dictionary of data frames in order to conduct recategorization and row dropping transformations at scale. We identified key variables within each year of data (the exact feature labels varied between years, so this reclassification would prove important for consistency across the entire dataset, and for further feature reduction and model operations). 
We wrote a few key cleaning functions - ‘recategorize,’ ‘count_ vals,’ and ‘drop_rows,’ which we applied across the sorted dictionary. We then created a copy of the dataframes, and dropped columns that were unrelated to our analytic objective (connecting physical activity and macronutrient composition to the occurrence of diabetes). Next, we renamed the columns (feature labels) across all of the dataframes, then added the years as a column to each of the data frames. We then concatenated the frames, creating a final data frame containing the data across all of the years of available data. 
We then removed correlated features, and any rows with missing values using the aforementioned ‘drop_rows’ function, applied to a new cumulative data frame called ‘result_cleaned.’ To view the percentage of removed values, we simply calculated the length of both data frames (number of rows) before and after cleaning, then subtracted and normalized this value to derive the efficacy of our cleaning and the completeness of our final dataset.
 
Methods

Final Preprocessing
The first phase in the process is data preprocessing, which marks the start of the procedure. Data preprocessing helps to clean, format, and organize the raw data by removing all the inconsistencies, inaccuracies, and missing data thereby making it ready for Machine Learning models. We acquired the Pima Indians Diabetes Database dataset during our initial research and then imported the relevant libraries. The libraries we imported that proved to be crucial were NumPy, Pandas, and Matplotlib. The next step was to detect and handle missing values effectively to avoid drawing incorrect conclusions and inferences from the data. This was accomplished by deleting a specific row with a null value for a feature or a specific column with more than 75% of the data missing, within our drop_row function. Splitting the dataset was the next step we completed. We split the data into a 80:20 ratio with taking 80% of the data for training the model and the balance 20% left for testing. Feature scaling was the final step in the pre-processing stage of this project. It was important to feature scale to limit the range of variables so that our group could compare them on the same level. We used the normalization method to do this. We used the sci-kit-learn library to help us do this. All the variables in the output are scaled between the values 0 and 1.

PCA
We used PCA to reduce the dimensionality of our dataset from 8 categories (not including the outcome data) and then condensing it to 7 categories of data. We then combined this data with the outcome column to create our final dataset. The reason we chose to do this is because by reducing the dataset with this method, we are coupling data that goes hand-in-hand in order to identify relationships in the data and helps maintain a standard throughout the dataset.

<img width="323" alt="image" src="https://user-images.githubusercontent.com/100390257/165456721-e7129dec-18c0-4976-aeb1-a5ba6bba6c08.png">

Logistic Regression:
Logistic Regression is used to analyze the data presented and return a probability value of how likely it is to get a certain outcome. In this case we are looking at the data and figuring out the probability of predicting diabetes. How we used this method was by fitting the training data of X to the training data of y and then predicting the class labels of X.

<img width="367" alt="image" src="https://user-images.githubusercontent.com/100390257/165457215-1cfeb389-c578-40f7-8283-ec25cb2d9b61.png">
 
Random Forest:
Random Forest was another method we tried, it was implemented similarly to Logistic Regression, but essentially what it does is turn the data into decision trees and makes its predictions based off of that. Because of this implementation Random Forest would probably work better on categorical data, which we do not have, so this may not be the most efficient way to predict diabetes. Aside from that, the more trees we create with this method the higher the accuracy should be so we tried it with 5, 10, 50, and 100 n_estimators in order to compare the results.

<img width="562" alt="image" src="https://user-images.githubusercontent.com/100390257/165457359-c34a0e87-17fa-43b3-a49b-b51a3bc7f1fa.png">

SVC:
The last method we implemented was the linear SVC method. This takes the data provided and finds a linear ‘best fit’ line that simplifies the data into one set of dependent and one set of independent data. This way the data is more easily analyzed as well as visualized. We implement this the same way as the last few methods as well by fitting the data and predicting the labels to find our accuracy.

<img width="320" alt="image" src="https://user-images.githubusercontent.com/100390257/165457480-51307367-926c-4b56-8519-41e05092531f.png">

Results & Discussion:
To obtain our results, we ran three different machine learning models: logistic regression, random forest, and support vector classification. They resulted in similar accuracies ranging between 70% to 80% with different ones doing slightly better than the other two each time the models were run. The random forest model, we varied the number of trees with four options: 5,10,50,100. They all performed similarly well on our dataset except 50 trees estimators performed slightly better than the other options. The graph of our results can be seen here.

<img width="474" alt="image" src="https://user-images.githubusercontent.com/100390257/165457622-82178efd-17f3-4cdb-9d45-95f8b699c260.png">

The accuracy of the other models, logistic regression and SVC,  can be seen here:

<img width="567" alt="image" src="https://user-images.githubusercontent.com/100390257/165457761-e56564c6-b7c6-4f15-a68d-8d7892b4af28.png">

From running our models, we were able to determine that the two most significant factors for a diabetes diagnosis are average number of pregnancies and average blood insulin levels. It’s important to note that these results are found from analysis of the Pima Indian Diabetes Dataset, and are representative of this female population.

<img width="540" alt="image" src="https://user-images.githubusercontent.com/100390257/165457940-b8476ab6-c08b-4792-8969-77f866e9e7f7.png">

To further upon what we already completed for the Midpoint Report, we decided to test our models on a larger dataset and so we switched our focus towards the NHANES dataset that we found during our initial research. We completed all the same steps we did for the Pima Indian Diabetes Dataset including cleaning the dataset, splitting the dataset, and feature scaling before running PCA to reduce the dimensionality and then implementing Logistic Regression and Random Forest on the dataset. After doing this we realized the conclusions we deduced using this dataset were different to those we found while using the initial dataset.

Random Forest:
<img width="528" alt="image" src="https://user-images.githubusercontent.com/100390257/165458421-42875269-d070-42ba-8fa0-5408e32a5903.png">

XGBoost:
<img width="530" alt="image" src="https://user-images.githubusercontent.com/100390257/165458511-9347b9d2-a5b1-404d-b219-d33d8e268157.png">

Future Work:
There are many ways to expand upon this project, such as taking into account even more data and variables that would have an influence on the prediction of diabetes. An even more elaborate project would include more studies and datasets along with the testing of more models and methods to see if they play a role in the output of our results. With further research and data in this area there is promise for more accurate predictive methods of diabetes.

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


