# Bike Sentiment Analysis

## Project Description
This project performs sentiment analysis on bike reviews using various machine learning and deep learning models. The goal is to classify bike reviews as positive or negative based on the review text. The project leverages state-of-the-art transformer models such as BERT and DistilBERT, as well as traditional machine learning models like Random Forest and Logistic Regression with TF-IDF features.

## Dataset
The dataset used is `Bike_Reviews_Final_1.csv`, which contains bike review descriptions and their corresponding sentiment labels (positive or negative). The dataset is preprocessed to handle missing values and balanced to have an equal number of positive and negative samples.

## Data Preprocessing
- Null values in review descriptions are replaced with empty strings.
- Null labels are replaced with -1 and such rows are dropped.
- The dataset is balanced by downsampling the majority class and oversampling the minority class to have 4000 samples each.
- The text data is tokenized and encoded using BERT and DistilBERT tokenizers with padding and truncation.
- Data is split into training, validation, and test sets with stratification to maintain label distribution.

## Models Implemented
1. **BERT-Base**: A BERT-based neural network with dropout and fully connected layers.
2. **DistilBERT**: A distilled version of BERT with a simpler architecture.
3. **Random Forest Classifier**: A traditional machine learning model using bagged decision trees.
4. **TF-IDF + Logistic Regression**: Logistic regression model trained on TF-IDF features extracted from the text.
5. **BERT-Large**: A larger BERT model with enhanced architecture and training setup.

## Training and Evaluation
- Models are trained using weighted loss functions to handle class imbalance.
- Training is performed for multiple epochs with evaluation on validation sets.
- The best model weights are saved based on validation loss.
- Final evaluation is done on the test set with classification reports showing precision, recall, and F1-score.

## Results and Visualizations
- Histograms of text length distributions.
- Bar charts and line plots comparing accuracy across different models.
- Radar charts visualizing model accuracy performance.

## Requirements
- Python 3.x
- PyTorch
- Transformers library by Hugging Face
- scikit-learn
- pandas
- numpy
- matplotlib

## How to Run
1. Install the required libraries:
   ```
   pip install torch transformers scikit-learn pandas numpy matplotlib
   ```
2. Place the dataset `Bike_Reviews_Final_1.csv` in the appropriate directory or update the path in the notebook.
3. Run the Jupyter notebook `BIKE_SENTIMENT_FINAL.ipynb` to execute the data preprocessing, model training, evaluation, and visualization steps.

## Notes
- The notebook includes multiple model implementations; you can select which model to train by changing the `model_choice` variable.
- GPU acceleration is used if available for faster training.
- The notebook saves the best model weights during training for later use.

---

This README provides an overview and instructions for the Bike Sentiment Analysis project based on the provided notebook.
