# Heart Disease Prediction

## 📌 Overview
Heart disease is one of the leading causes of mortality worldwide. This project aims to develop a machine learning model to predict heart disease based on various medical and lifestyle parameters.

## 🏗 Project Structure
```
📂 Heart-Disease-Prediction
│── 📄 README.md               # Project documentation
│── 📂 dataset                 # Contains dataset files
│── 📂 models                  # Trained ML models
│── 📂 notebooks               # Jupyter notebooks for EDA and model training
│── 📂 scripts                 # Python scripts for data preprocessing & model training
│── 📂 results                 # Model evaluation results & visualizations
│── 📜 requirements.txt        # Python dependencies
│── 📜 .gitignore              # Files to be ignored in Git tracking
```

## 📊 Dataset
The dataset used for this project contains various health metrics such as:
- Age
- Sex
- Blood Pressure
- Cholesterol Levels
- Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression

Dataset source: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## ⚙️ Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Models**: Logistic Regression, Random Forest, Support Vector Machine (SVM)
- **Visualization Tools**: Matplotlib, Seaborn
- **Jupyter Notebook** for development

## 📌 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/KK-coder-12/Heart-Disease-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## 🚀 Model Training & Evaluation
1. **Data Preprocessing**: Handle missing values, normalize data, and encode categorical variables.
2. **Feature Engineering**: Select important features using correlation analysis.
3. **Model Training**: Train multiple ML models and fine-tune hyperparameters.
4. **Evaluation**: Compare accuracy, precision, recall, and F1-score.

## 🔍 Results
- The **Random Forest model** achieved the highest accuracy of 85%.
- Feature importance analysis showed that **cholesterol levels and blood pressure** significantly impact heart disease prediction.

## 📌 Future Enhancements
- Improve feature selection techniques
- Deploy the model as a web application
- Implement deep learning models for better accuracy

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests.

## 📜 License
This project is licensed under the **MIT License**.


