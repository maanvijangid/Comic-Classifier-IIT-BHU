⭐ Cosmic Classifier: Machine Learning Approach for Celestial Object Classification

📚 Overview
This project builds a complete machine learning pipeline to classify celestial objects  based on their physical and environmental properties — things like atmospheric density, gravity, surface temperature, radiation levels, etc.

📁 Dataset
The dataset had missing values and outliers.
There was a class imbalance problem, meaning some celestial categories were underrepresented.
Key features:
Numerical columns: Atmospheric density, Surface temperature, Gravity, Water content, Mineral abundance, Orbital period, Proximity to star, Atmospheric composition index, etc.
Categorical columns: Magnetic field strength and Radiation levels (converted into numerical form by parsing category names).
Sequential proceedings followed by the team in creating the model are as follows:

STEP1:EDA and Preprocessing 
a) We began by checking for class imbalance through .value_counts() function.We noticed moderate class imbalance between the various classes present with the least occurring class(5.0) having 5132 samples and highest occurring class(1.0) having 6393 samples.

b) Before handling the class imbalance issues,we further checked for outliers through boxplots of the columns.We found outliers in numerous columns like 'Gravity','Surface Temperature' etc.

c) Then we proceeded to check for percentage of missing values in the columns if any and findings showed approximately 5% missing values for each of the columns.

d)We decided to drop rows containing missing values in the Prediction column(which was the target column) as it had close to 5% missing values and we didn't want to affect target column values in any way through imputation so resorted to removing the rows.KNN imputation was applied to the rest of the columns(including columns 'Magnetic Field Strength' and 'Radiation Levels' which were converted to numeric type by extracting their numerical parts).KNN was chosen as dataset is relatively small.

e) Winsorization technique was applied for outlier treatment instead of removal of rows as removal of rows could have detrimental impacts on the dataset structure and lead to wrong outcomes.

f)We then checked if there were any correlations between the columns but wasn't able to find any strong correlation as such.

g)Then we resorted to use SMOTE for handling class imbalance even though the sample difference between least and most frequent classes wasn't very vast.The reason being it gave better performance in view of other ways of class imbalance handling(which we had already tried) such as random oversampling,undersampling etc.

STEP 2:TRAINING AND EVALUATION

🤖 Models Used
Two main classifiers:
Random Forest and XGBoost which were used to build an ensemble model using VotingClassifier to combine both models for more robust performance.

⚙️ Hyperparameter Tuning
Used GridSearchCV with cross-validation to optimize model parameters.

Explanation: Balanced choices to avoid underfitting/overfitting and keep computation reasonable.

🧩 Why GridSearchCV?
Exhaustively checks all combinations of hyperparameters to find best combination of hyperparameters.
Uses cross-validation to avoid overfitting.
Automatic parallel processing.
Alternatively RandomizedSearchCV considered but not used since Random Search CV randomly selects combinations meaning there is no guarantee of finding the absolute best set of hyperparameters.

📊 Model Performance
The ensemble performed well and we were able to finally get an accuracy of 0.9135 as well as f1 score of 0.9134.

STEP 3:MODEL PREDICTION ON INPUT TEST SET

We then proceeded to provide the code in the notebook for running prediction on input datsets using the ensemble learning model we built.

✅ Project Takeaway
Showcased the effectiveness of ensemble learning and robust data preprocessing in astrophysical object classification.
Identified key features and tackled common challenges: outliers, missing values, and class imbalance.
Provided a solid foundation for future work or scaling up to more advanced deep learning models.


