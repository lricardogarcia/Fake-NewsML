# Fake News Prediction using Machine Learning

This project uses Machine Learning to classify news articles as **Real** or **Fake**. The model is trained on a labeled dataset of news articles and uses **Logistic Regression** for the prediction task.

## Project Overview

The objective of this project is to detect fake news by performing text preprocessing, feature extraction, and building a machine learning model that predicts whether a news article is real or fake.

## Dataset

The dataset contains news articles labeled as **Real** or **Fake**, along with additional information like the author and title of the article. The dataset is processed to create a feature vector that helps the machine learning model make accurate predictions.

- You can download the dataset from this [Google Drive link](https://drive.google.com/drive/folders/19pokFUDzCaF4d7nnKsQlU09L2EotaKh0?usp=sharing).

## Dependencies

The following libraries are required to run this project:

- `numpy`
- `pandas`
- `nltk` (Natural Language Toolkit)
- `scikit-learn`
- `re` (regular expressions)

## Installation

Make sure to also download the NLTK stopwords by running:
````bash
import nltk
nltk.download('stopwords')
````

Install the dependencies:

```bash
  pip install -r requirements.txt
```
    
## Project Workflow
- **Data Preprocessing:** The dataset is loaded into a pandas DataFrame. Missing values are handled by replacing null values with empty strings. The author and title fields are merged to create a single text column called content, which will be used for the analysis. Stemming is applied to the content column to reduce words to their root forms (e.g., "running" becomes "run").
- **Feature Extraction:** The textual data is converted to numerical features using TF-IDF Vectorization.TF-IDF (Term Frequency-Inverse Document Frequency) helps represent text data in numerical format by considering the importance of words in the dataset.
- **Model Training:** The dataset is split into training and testing sets using train_test_split with 80% of the data used for training and 20% for testing. A Logistic Regression model is used for training on the processed data. The model is then evaluated on both the training and testing datasets using accuracy score.
- **Making Predictions:** A predictive system is built to classify new news articles as either Real or Fake. After processing new input data, the model predicts the label (0 for Real, 1 for Fake).

## Model Accuracy
The model is evaluated using accuracy scores on both the training and test datasets:
- **Training data accuracy:** X%
- **Test data accuracy:** Y%

## Detailed Steps
- **Stemming:** We apply stemming to reduce words to their root forms. This helps in reducing the dimensionality of the text data and improving the model's generalization capability.
- **TF-IDF Vectorization:** TF-IDF is used to transform the textual data into a numerical format, which is essential for the logistic regression model.
- **Logistic Regression:** We use Logistic Regression, a simple yet effective model for binary classification problems such as real vs. fake news.
- **Evaluation:** The accuracy of the model is measured using the accuracy_score metric, providing insights into how well the model performs on unseen test data.
## License

This project is licensed under the MIT License.

### Key Features:
- **Dataset**: The dataset contains real and fake news labeled for classification tasks.
- **Model**: Logistic Regression is chosen for this binary classification problem.
- **Accuracy**: Accuracy is evaluated on both training and test data.
