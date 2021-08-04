# Predicting driver's alertness using machine learning algorithm

## **The Goal**

The goal is to create a model to classify whether the driver is being alert or not while driving, using the data acquired from multiple controlled experiments by Ford.
![image](https://user-images.githubusercontent.com/80243823/117619770-3fe93b00-b1a2-11eb-8c73-28a1f4d54795.png)


## **The raw data**
We have 33 anonymous variables with P = physiological data, E = environmental data, V = vehicular data. Target is to classify the variable 'IsAlert'. There are in total 604,329 data entries.
![image](https://user-images.githubusercontent.com/80243823/117620710-52b03f80-b1a3-11eb-9bbd-7177012d1a0f.png)
![image](https://user-images.githubusercontent.com/80243823/117620843-7a070c80-b1a3-11eb-880d-519c061b473b.png)


## **Data preprocessing**
We first check the data type, number of missing values and the existence of possible outliers within the data set
![image](https://user-images.githubusercontent.com/80243823/117621250-fd286280-b1a3-11eb-847f-f0c3362ba159.png)

We then drop the useless (contains value '0' only) variables and drop those with low correlation with target variable 'IsAlert'
![image](https://user-images.githubusercontent.com/80243823/117622110-de769b80-b1a4-11eb-92b2-ff048bf719a1.png)

With over 20 variables at hand, we apply 'Principal Component Analysis' (PCA) in hope to reduce the complexity of the model. Our aim is to find enough variables to explain over 95% of the variance. We have found that having 20 components would be enough.

![image](https://user-images.githubusercontent.com/80243823/117622786-9c018e80-b1a5-11eb-9c13-ad8f0d22551b.png)

After extensive testing, we find that the improvement of applying PCA is not significant for this dataset. The testing results can be found in 'Performance Comparison' section, which will be shown later.

## **Models evaluation and comparison**
We evaluate 4 possible models, Logistic Regression, KNN, Random Forest and XGBoost
![image](https://user-images.githubusercontent.com/80243823/117623836-dae41400-b1a6-11eb-89ae-5cdad32c018e.png)


**Logistic Regression**

The target performance metric we are looking for is the recall rate of the value 0, that is how accurate we can detect if the driver is being not alert.
On the other hand, if the driver is being alert, that is the value 1, we don't mind having a lower accuracy.
![image](https://user-images.githubusercontent.com/80243823/117625528-b0935600-b1a8-11eb-95ba-7b7cdc80b967.png)

69% accuracy is the performance of the initial model. We then use 'GridSearchCV' to determine how strict we want to limit the model in order to give the best result.
![image](https://user-images.githubusercontent.com/80243823/117626084-4f1fb700-b1a9-11eb-9dd7-e6992bcd25de.png)

Unfortunately, even applying the best parameter calculated by the model, the improvement is minimal.
![image](https://user-images.githubusercontent.com/80243823/117626137-6068c380-b1a9-11eb-81c3-994c39cf17c2.png)

We then further reduce the collinearity by dropping more variables. No matter what combination of variables we use, 71% recall rate seems to be the limit of Logistic Regression.
![image](https://user-images.githubusercontent.com/80243823/117626833-1d5b2000-b1aa-11eb-9d3d-0816f7bb5a4e.png)


**KNN**

This is a list of recall rates using different values of K. With K = 1, that is looking at the closest data point as the answer, we could achieve 97% accuracy in the training dataset. However, this is clearly model overfitting. Its performance is going to worsen in testing dataset, which will be shown later.

![image](https://user-images.githubusercontent.com/80243823/117629515-d1f64100-b1ac-11eb-9336-2fd8b29f6671.png)


**Random Forest**

We first train the model using default parameters, then tune them using 'GridSearchCV'. It has 83% and 97% recall rate respectively.
![image](https://user-images.githubusercontent.com/80243823/117629934-3dd8a980-b1ad-11eb-838f-513367796cff.png)

To avoid overfitting, we also evaluate the performance differences between training and testing dataset. We have found that number of estimators has minimal impact on the recall rate, while having max_depth = 20 would yield satisfactory recall rate and small performance differences between training and testing dataset.
![image](https://user-images.githubusercontent.com/80243823/117630243-9314bb00-b1ad-11eb-89be-1a4d07dc2f64.png)

This is the final model after applying the optimal parameters.
![image](https://user-images.githubusercontent.com/80243823/117631930-55189680-b1af-11eb-855a-197dc2a880b2.png)
  

**XGBoost**

We apply the same training and tuning procedures as Random Forest to XGBosst
![image](https://user-images.githubusercontent.com/80243823/117631322-b2f8ae80-b1ae-11eb-92dd-fa0c7fad0532.png)
![image](https://user-images.githubusercontent.com/80243823/117631758-23073480-b1af-11eb-8122-1954aaf06ca1.png)
![image](https://user-images.githubusercontent.com/80243823/117631865-3f0ad600-b1af-11eb-952f-5d2496786f38.png)

**Performance Comparison**

To measure the performance of different models under different conditions, we plot the 'Receiver Operating Characteristic Curve' (ROC curve). We want to select the model which has higher true positive rate and lower false positive rate under different threshold (upper left corner).
![image](https://user-images.githubusercontent.com/80243823/117632093-7c6f6380-b1af-11eb-8208-bc7cfeb7e690.png)

To sum up using 1 number, we calculate the 'Area Under Curve' (AUC) of the ROC curve. The higher the better.
![image](https://user-images.githubusercontent.com/80243823/117634946-fef92280-b1b1-11eb-98f4-6641495ea25e.png)


## **Conclusion**
Random Forest gives us the best overall performance, with 94% accuracy to predict the driver not being alert (under threshold = 0.5) and 78% AUC in the testing set.
However, the degradation of performance from training dataset to testing dataset suggests that the model is still overfitting, so there is still room for improvement. Instead of eliminating useless variables one by one, next approach is to add useful variables one by one.
![image](https://user-images.githubusercontent.com/80243823/117636337-49c76a00-b1b3-11eb-9fde-5a6f70670f58.png)
