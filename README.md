# smartphone-price-prediction
 
# Introduction:

With the increasing demand for smartphones, mobile phone manufacturers are constantly striving to release new models with improved features, performance, and design. As a result, the mobile phone market has become highly competitive, with a wide range of options available at various price points. In this project, we will explore the following dataset from Kaggle: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?datasetId=11167&sortBy=voteCount&language=Python that contains information about the specifications and prices of various mobile phones.

The aim of this project is to develop a machine learning model that can predict the price range of a mobile phone based on its specifications. We will use a supervised learning approach to train the model using a labeled dataset. The dataset contains information about 20 different features for each mobile phone, including battery capacity, camera quality, screen size, and more. The target variable is the price range, which is divided into four categories: low-cost, medium-cost, high-cost, and very high-cost.

Various data science libraries such as pandas, numpy, scikit-learn and more, were used to visualize and preprocess the data, and then build various machine learning models by training them on the training dataset. We will choose the model with the highest accuracy to predict the values for the testing dataset.

Overall, this project will provide valuable insights into the factors that influence the price range of mobile phones and will demonstrate the effectiveness of machine learning in predicting these price ranges. This will allow manufacturers to better price their mobile phones according to the current market standards and also make informed decisions.

# Methodology

A supervised learning approach was used to train various machine learning models. The dataset was first checked for missing values to ensure no data was absent. The training dataset was further split into training and testing sets using an 80:20 ratio respectively. Then, various classification algorithms such as logistic regression, k-nearest neighbors, decision trees, and random forest were used to train the model on the training set. The performance was then evaluated various metrics such as accuracy, precision, recall, and F1-score. The results were visualized using a Confusion Matrix. The performance of the models used was then compared to choose the best performing model. This was used to predict price range for the unseen data (test.csv).

## Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest
- KNN
- Support Vector Machine


# Conclusion
Based on the results obtained, we can conclude that the Logistic Regression model performed the best, with the highest accuracy, precision, recall, and F1 score. Thus, this model can be considered as the best predictor of price range for mobile phones based on their features in this dataset. We can also conclude that if more unseen data, consisting of the same features, was presented to the model we could expect it to perform just as good and produce accurate results. This can help mobile manufacturing companies in making data-backed informed decisions about pricing their mobile devices.