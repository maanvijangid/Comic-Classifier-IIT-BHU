⭐ Cosmic Classifier: Machine Learning Approach for Celestial Object Classification

📚 Overview
This project builds a complete machine learning pipeline to classify celestial objects (like planets, stars, etc.) based on their physical and environmental properties — things like atmospheric density, gravity, surface temperature, radiation levels, etc.

📁 Dataset
The dataset had missing values and outliers.
There was a class imbalance problem, meaning some celestial categories were underrepresented.
Key features:
Numerical columns: Atmospheric density, Surface temperature, Gravity, Water content, Mineral abundance, Orbital period, Proximity to star, Atmospheric composition index, etc.
Categorical columns: Magnetic field strength and Radiation levels (converted into numerical form by parsing category names).

🔎 Preprocessing Done
Missing values handled with KNNImputer (using nearest neighbors to fill missing data).
Outliers detected using IQR and treated using Winsorization (capping extreme values).
Correlation analysis showed no strong linear relationships between features.
Class imbalance handled with SMOTE (synthetic oversampling).

🤖 Models Used
Two main classifiers:
Random Forest
XGBoost
Built an ensemble model using VotingClassifier to combine both models for more robust performance.

⚙️ Hyperparameter Tuning
Used GridSearchCV with cross-validation to optimize model parameters.
Key hyperparameters tuned:
n_estimators, max_depth, min_samples_split for Random Forest.
n_estimators, max_depth, learning_rate for XGBoost.
Explanation: Balanced choices to avoid underfitting/overfitting and keep computation reasonable.

🧩 Why GridSearchCV?
Exhaustively checks all combinations of hyperparameters.
Uses cross-validation to avoid overfitting.
Automatic parallel processing.
Alternatives (like RandomizedSearchCV, Bayesian Optimization, manual tuning) were considered but not used.

📊 Model Performance
The ensemble performed well, but a neural network was also tested and achieved 86% accuracy — indicating potential for improvement through deep learning.
The tree-based ensemble is more interpretable but might not capture all complex patterns.
And finally we get accuracy of 0.91

✅ Project Takeaway
Showcased the effectiveness of ensemble learning and robust data preprocessing in astrophysical object classification.
Identified key features and tackled common challenges: outliers, missing values, and class imbalance.
Provided a solid foundation for future work or scaling up to more advanced deep learning models.


