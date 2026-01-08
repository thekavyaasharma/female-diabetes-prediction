# 🩺 Female Diabetes Prediction Using SVC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Support%20Vector%20Classification-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-77.9%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*Leveraging machine learning to predict diabetes risk in female patients with 78% accuracy*

[Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) • [Report Issue](../../issues) • [Request Feature](../../issues)

</div>

---

## 🎯 About The Project

Early diabetes detection can be life-changing. This project harnesses the power of **Support Vector Classification (SVC)** to predict diabetes risk in female patients based on clinical measurements. Trained exclusively on the renowned Pima Indians Diabetes Dataset, our model achieves impressive accuracy while maintaining excellent generalization.

### ✨ Key Highlights

-  **77.9% Accuracy** on unseen test data
-  **7 Clinical Features** for comprehensive assessment
-  **No Overfitting** - Model generalizes beautifully
-  **Real-time Predictions** with custom predictive system
-  **Clean Dataset** - Ready for immediate analysis

---

## 📊 Dataset

We utilize the **Pima Indians Diabetes Database**, a landmark dataset containing diagnostic measurements from female patients of Pima Indian heritage aged 21 years or older.

**📁 Source:** [Kaggle - UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Why this dataset?** The Pima Indians population has one of the highest incidences of diabetes worldwide, making this dataset particularly valuable for medical research and predictive modeling.

---

## 📁 Repository Structure

```
📦 female-diabetes-prediction
├── 📓 DiabetesPrediction.ipynb    # Complete ML pipeline & implementation
├── 📋 data_dictionary             # Feature descriptions & metadata
├── 📄 requirements.txt            # Python dependencies
└── 📊 diabetes.csv                # Training & testing data
```

---

## 🔬 Features Used for Prediction

Our model analyzes **7 critical health indicators**:

| Feature | Description | Clinical Significance |
|---------|-------------|----------------------|
|  **Pregnancies** | Number of times pregnant | Gestational diabetes risk factor |
|  **Glucose** | Plasma glucose concentration | Primary diabetes indicator |
|  **Blood Pressure** | Diastolic blood pressure (mm Hg) | Cardiovascular health marker |
|  **Skin Thickness** | Triceps skin fold thickness (mm) | Body fat distribution indicator |
|  **Insulin** | 2-Hour serum insulin (mu U/ml) | Insulin resistance assessment |
|  **BMI** | Body mass index (weight/height²) | Obesity indicator |
|  **Diabetes Pedigree** | Genetic predisposition function | Family history impact |
|  **Age** | Age in years | Risk increases with age |

---

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip >= 20.0
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/female-diabetes-prediction.git
   cd female-diabetes-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook DiabetesPrediction.ipynb
   ```

4. **Run all cells** and start making predictions! 🎉

---

## 🔄 ML Pipeline

Our comprehensive machine learning workflow:

1. **Data Loading** - Import the cleaned Pima Indians dataset
2. **Exploratory Data Analysis** - Uncover patterns and correlations
3. **Feature Preparation** - Separate features (X) and target variable (y)
4. **Standardization** - Scale features using StandardScaler (μ=0, σ=1)
5. **Data Splitting** - 80-20 train-test split for robust evaluation
6. **Model Training** - Train SVC classifier on training data
7. **Performance Evaluation** - Assess using accuracy and precision metrics
8. **Predictive System** - Deploy model for real-world predictions

---

## 📈 Model Performance

### 🎯 Metrics Overview

| Dataset | Accuracy | Precision | Interpretation |
|---------|----------|-----------|----------------|
| **Training** | 78.0% | 74.5% | Strong baseline performance |
| **Testing** | 77.9% | 75.0% | Excellent generalization ✅ |

### 💡 Key Insights

- **Minimal Performance Gap** - Only 0.1% difference between training and testing accuracy
- **No Overfitting** - Model generalizes exceptionally well to unseen data
- **Balanced Precision** - 75% precision ensures reliable positive predictions
- **Production-Ready** - Consistent performance indicates deployment readiness

### 📊 Visual Performance

```
Training Accuracy:   ████████████████░░░░  78.0%
Testing Accuracy:    ████████████████░░░░  77.9%

Training Precision:  ███████████████░░░░░  74.5%
Testing Precision:   ███████████████░░░░░  75.0%
```

---

## 🔮 Making Predictions

Use our intuitive predictive system to assess diabetes risk:

```python
# Example: Predict for a new patient
patient_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
prediction = model.predict(scaler.transform([patient_data]))

if prediction[0] == :
    print("✅ Low diabetes risk")
else:
    print("⚠️ High diabetes risk detected")
```

---

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Scikit-learn** - SVC implementation & preprocessing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

---

## 📚 Learning Outcomes

Working with this project, you'll gain experience in:

- ✅ Binary classification with Support Vector Machines
- ✅ Feature standardization and preprocessing
- ✅ Train-test split methodology
- ✅ Model evaluation using multiple metrics
- ✅ Handling medical datasets responsibly
- ✅ Building end-to-end ML pipelines

---

## 🤝 Contributing

Contributions make the open-source community amazing! Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** - For providing the dataset
- **National Institute of Diabetes and Digestive and Kidney Diseases** - Original data collection
- **Kaggle Community** - For maintaining and sharing the dataset
- **Scikit-learn Contributors** - For the excellent ML library

---

## 📬 Contact

Have questions or suggestions? Feel free to reach out!

**Project Link:** [https://github.com/thekavyaasharma/female-diabetes-prediction](https://github.com/thekavyaasharma/female-diabetes-prediction)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with Python

</div>
