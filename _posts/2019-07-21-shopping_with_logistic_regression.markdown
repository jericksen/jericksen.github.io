---
layout: post
title:      "Shopping With Logistic Regression"
date:       2019-07-21 17:22:02 +0000
permalink:  shopping_with_logistic_regression
---


This blog was published on Medium. Visit the original post using the link below:

https://medium.com/@jericksen20/shopping-with-logistic-regression-be7db7f804a
---

I found a dataset online that was a record of visitor behavior to an eCommerce website. As a budding data scientist, I thought it would be interesting to attempt replication of some of the website behavior one sees when shopping online. Specifically, as it pertains to the 'special offers' and 'act now and you'll receive a 15% discount' pop-ups promotions.  
In the age of data, few things in the online shopping arena are left to chance or generosity. These 'special offers' are no doubt driven by machine learning algorithms working hard to predict your propensity to transact. Should an algorithm recognize you as a returning customer with a hefty wishlist and a number of visits to a particular product page (we'll use a pair of red shoes in this example), then you deserve a 10% discount to help close the deal. On the flip side, if you're a heavy spender that spends a few minutes shopping before clicking 'confirm your order', then we, the company, will save that 10% discount and offer it to the next wide-eyed red shoe shopper. 
Here's the point, your online behavior is readable. Given enough data, engineers can use visitor data to begin aggregating and modeling human shopping behavior. Once we have our finely tuned model with enough data, we know enough about you to predict whether or not you're going to purchase those red shoes. 
As someone fascinated by machine learning, and one just starting out as a novice data scientist, I thought I'd attempt to recreate one of these algorithms that predicts your next transaction. So with that, let's dive into a brief overview of how I approached this neat project. 


---

There are a great number of machine learning algorithms used as classifiers, support vector machines, random forests, and decision trees to name a few. For this project, I chose to go with logistic regression. 
Logistic regression models are similar to linear regression models with one twist, they use the sigmoid function. The sigmoid function takes what would otherwise be a linear model that best fits a given dataset and shapes it into an S curve that can be thought of as a probability curve. As new data is plotted, it is assigned a number between 0 and 1. The closer the number is to 1, the higher the probability that it can be classified as a 1 and vice versa for 0. 
It's quite fascinating: we take all the regressors and sandwich them into a figure between 0 and 1. We set our threshold, often at .5, and use this as our final decision maker on whether the event is a class 1 or 0. Crazy. 
Below is a run-through of the steps used to implement a logistic regression using shoppers data. Of course, not all the intricacies of this project are included, but enough to give the average reader a soft introduction of what goes on behind the scenes with machine learning algorithms. 


---

Import the Data: 
df = pd.read_csv('OnlineShopperIntention.csv')
display(df.head())
display(df.tail())
Perform a number of EDA practices. Below is an example of a plot used to visualize the scale of the numerical features: 
num_attributes.hist(figsize=(20,20))
plt.show()
Plot a correlation matrix to assess and remove collinearity among feature variables: 
corr = df.corr()
plt.figure(figsize=(15,13))
sns.heatmap(corr, fmt='.2g', cmap = 'Blues', annot = True, linewidth = 2, robust = True)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.show()
Rebalance the data set using the SMOTE method from SKlearn: 
x = feats
y = target
print(y.value_counts())
x_resampled, y_resampled = SMOTE().fit_sample(x,y)
print(pd.Series(y_resampled).value_counts())
scaled_feats = pd.DataFrame(x_resampled, columns = feats.columns)
scaled_target = y_resampled
Normalize the data: 
names = feats.columns
scaler = StandardScaler()
scaled_df = scaler.fit_transform(feats)
scaled_feats = pd.DataFrame(scaled_df, columns = names)
scaled_feats.head()
Run a grid search to find the optimal parameters using accuracy as our scoring metric:
x_train, x_test, y_train, y_test = train_test_split(scaled_feats, scaled_target, test_size = .20)
grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], 'fit_intercept':['True', 'False'], 
        'class_weight': ['balanced', None]}
logreg = LogisticRegression(random_state=15)
logreg_cv = GridSearchCV(logreg,grid,cv=3)
logreg_cv.fit(x_train, y_train)
print("Tuned Hyperparameters: ",logreg_cv.best_params_)
print("Accuracy: ",logreg_cv.best_score_)
Fit our model:
logreg = LogisticRegression(random_state=11)
model_log = logreg.fit(x_train, y_train)
Make some predictions with the fitted model:
y_hat_train = logreg.predict(x_train)
residuals = np.abs(y_train - y_hat_train)
print('Model Predictions:')
print()
print(pd.Series(residuals).value_counts())
print()
print(pd.Series(residuals).value_counts(normalize=True))
Print the model's scoring metrics:
print('Recall: {}'.format(recall_score(y_train, y_hat_train)))
print('Precision: {}'.format(precision_score(y_train, y_hat_train))) 
print('Accuracy: {}'.format(accuracy_score(y_train, y_hat_train)))
Build the resulting confusion matrix from our test set: 
cnf_matrix = confusion_matrix(y_hat_test, y_test)
print('Confusion Matrix:\n',cnf_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap= 'Blues', linewidths=10, )
plt.title('Confusion Matrix', size= 15)
plt.xlabel('Predicted Classes', size = 13)
plt.ylabel('Actual Classes', size = 13);
Assess the Reciever Operating Curve (ROC) and Area Under the Curve (AUC) for both the train and test set: 
y_test_score = logreg.decision_function(x_test)
y_train_score = logreg.decision_function(x_train)
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_score)
print('Initial Model Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('Initial Model Train AUC: {}'.format(auc(train_fpr, train_tpr)))
plt.figure(figsize=(10,8))
plt.plot(test_fpr, test_tpr, color = 'blue', lw = 2, label ='Initial Model Test ROC curve')
plt.plot(train_fpr, train_tpr, color = 'purple', lw = 2, label ='Initial Model Train ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate', size = 12)
plt.ylabel('True Positive Rate', size = 12)
plt.title('Receiver operating characteristic (ROC) Curve', size = 14)
plt.legend(loc="lower right");
And finally, we interpret the results: The model 
Accuracy
The test and train set returned similar a similar accuracy score indicating no over fitting on the initial model.
The accuracy scores were in the high 80%'s for both the train and test set which is a relatively decent first model. However, the highly imbalanced nature of the data set may be influencing the performance metrics, particularly with respect to making positive predictions.

Precision
Our precision score for both the train and test set were relatively good with with a score of 73% and 75% respectively.

Recall
Our recall scores are poor, however given the business case, we're not going to focus on these scores as precision is the result for which we hope to optimize.

AUC, Threshold
With an AUC of roughly 90 percent, we can be confident our model is making the correct prediction 90 percent of the time. These scores are pretty good.



---

Classifiers, like logistic regression, are used frequently to make predictions. And they are no doubt used frequently within your favorite online shopping environment to guide decisions on when to offer you promotions. I hope the above example is enlightening to some and helpful to others!
