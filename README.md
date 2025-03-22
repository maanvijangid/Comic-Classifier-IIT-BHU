# ⭐ Cosmic Classifier: Machine Learning Approach for Celestial Object Classification

---

## 📚 **Overview**  
This project builds a complete machine learning pipeline to classify celestial objects based on their physical and environmental properties — things like atmospheric density, gravity, surface temperature, radiation levels, etc.

---

## 📁 **Dataset**  
The dataset had missing values and outliers. There was a class imbalance problem, meaning some celestial categories were underrepresented.

**Key features:**  
- **Numerical columns:** Atmospheric density, Surface temperature, Gravity, Water content, Mineral abundance, Orbital period, Proximity to star, Atmospheric composition index, etc.  
- **Categorical columns:** Magnetic field strength and Radiation levels (converted into numerical form by parsing category names).  

---

## 🛠️ **STEP 1: EDA and Preprocessing**  
- Checked for class imbalance using `.value_counts()`.  
   - Least occurring class (5.0) had 5132 samples and the most frequent class (1.0) had 6393 samples.  
- Outliers were identified using boxplots in columns like **Gravity** and **Surface Temperature**.  
- Found approximately **5% missing values** for each column.  
- Rows with missing values in the **Prediction** (target) column were dropped.  
- Applied **KNN imputation** for other missing values (including numeric conversion of **Magnetic Field Strength** and **Radiation Levels**).  
- Outliers were treated using **Winsorization** instead of removing rows.  
- Checked for feature correlation, but no strong correlation was found.  
- Applied **SMOTE** to handle class imbalance after testing other techniques like random oversampling and undersampling, as SMOTE yielded better results.  

---

## 🤖 **STEP 2: Training and Evaluation**  

### **Models Used:**  
- Random Forest  
- XGBoost  
- Both combined using an **ensemble model with VotingClassifier** for robust performance.  

### **Hyperparameter Tuning:**  
- Used **GridSearchCV** with cross-validation.  

**Why GridSearchCV?**  
- Exhaustive search across hyperparameter combinations.  
- Uses cross-validation to avoid overfitting.  
- Supports parallel processing.  
- Chosen over RandomizedSearchCV for guaranteed optimal results.  

### **Model Performance:**  
- **Accuracy:** 0.9135  
- **F1 Score:** 0.9134  

---

## 🧪 **STEP 3: Model Prediction on Input Test Set**  
- Code provided in the notebook to run predictions on input datasets using the trained ensemble model.  

---

## ✅ **Project Takeaway**  
- Showcased the effectiveness of ensemble learning and robust data preprocessing in astrophysical object classification.  
- Successfully handled outliers, missing values, and class imbalance.  
- Identified key features contributing to classification performance.  
- Set a strong foundation for future work, potentially scaling to deep learning models.  


