---
layout: post
title:      "DC Crime Analysis"
date:       2019-09-17 00:16:46 +0000
permalink:  dc_crime_analysis
---

The original blog post can be found on Medium: https://medium.com/@jericksen20/dc-crime-analysis-60f10f68e7f7

---

For a recent project, I elected to dive into the crime data for Washington, DC available to the public through the DC Open Data initiative. As a DC resident, I thought it would be interesting to map crime incidents throughout the city as well as analyze crime volumes via a few different dimensions. Further, I thought it would be fun to attempt building a classifier that predicted the type of crime to have taken place given a few geographical and time features. 
I'll briefly walk through the methods used to create a few neat visualizations as well as my approach to the modeling portion of the project. I'll end with some final thoughts as well as some items needed to improve the model's performance. 


---

Exploratory data analysis (EDA) is often my favorite part of any data science project. I find visualizing data to be a fun task, particularly with the right tools. For this project, I chose to work with the application Tableau to create some of the visuals as well as the python mapping library folium. 
To begin, I looked at the breakdown of crime by the differing crime categories as defined by the DC police department. I focused my EDA analysis on the most recent 12-month cycle at the time of this project which ended July 31st, 2019: 
DC Crime by TypeI then looked at the volume of crime parsed by month. The intent was to extract any seasonality that occurs in terms of crime activity: 

DC Crime by Month
The image above outlines a clear seasonal cycle with late summer/early fall being peak months and late winter/early spring representing the quieter months. 
I did the same with the hour of day. Again looking for any daily cycles in terms of crime activity:

DC Crime by Hour
Once again, a clear structure was indicated by the visualization with early morning hours remaining relatively quiet compared to early afternoon and late evenings. 
DC Crime - HeatmapNext, I looked at the geographical distribution of crime throughout the city using a heatmap. The data used came from the same 12-month period as defined above. 
Some interesting characteristics stood out from the resulting map. The highest concentration of crime existed near the downtown area. Beyond that, pockets of higher crime rates appeared to exist near metro stations. Rock Creek Park, unsurprisingly, showed no crime incidents. And last, the Northwest quadrant generally showed fewer incidents of crime vs. Northeast & Southeast. For those familiar with the socio-economic footprint of DC, this is perhaps unsurprising as it's generally known that Northeast/Southeast house lower-income families with the Northwest quadrant includes wealthier residents. 
DC WardsThe last EDA visual to highlight is the breakdown of crime by DC ward. There are 8 wards in DC and are separated as the image to the left depicts.
Below, I've included the analysis of each ward. Ward two accounts for 1/5 of all crime. It also has the highest crime rate of 7.5% compared to the average rate of 4.8%. Unsurprisingly, wards 3 and 4 have the lowest crime rates with wards 5 and 6 having above-average rates. 



---

With the EDA summary complete, I'll discuss the process for generating a classification model that predicts crime types. Inevitably, when tasked with a classification problem, the issue of choosing the right model is bound to be a sticking point. To solve for this, I chose to run multiple classifiers and compare the results using their cross-validation scores. Doing so highlighted the best performing model from which I further improved using a SKlearn's grid search method. 
With the data cleaned, rebalanced, and normalized, the next step was to generate a test and train set from the data set. I did so using the train_test_split method from SKlearn:
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = .20)
With the training and testing set split, I created a list object which housed the all of the instantiated models chosen for this test:
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('ADB', AdaBoostClassifier()))
From there, I used a for-loop to iterate through each model in my list to fit the models on a subsection of the training set. I used the accuracy scores from the KFolds cross-validation method when comparing each model: 
seed = 42
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    
    kfold = model_selection.KFold
    (n_splits = 20, random_state = seed)
    
    cv_results = model_selection.cross_val_score(model, 
    X_train, y_train, cv = kfold, scoring = scoring)
    
    results.append(cv_results)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    
    print(msg)
The resulting output from the for-loop included the average accuracy score from each fold of data along with the standard deviation from the 20 samples: 
Model Avg. Accuracy + Standard Deviation (20 samples)I then plotted the results for a visual reference: 
plt.figure(figsize=(8,8))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison - CV Accuracy Scores', size = 15)
plt.xlabel('Classifier', size = 13)
plt.ylabel('Accuracy', size = 13)
plt.xticks(size = 13)
plt.yticks(size = 13)
plt.show()
With the best model highlighted above, the next step was to attempt to further improve the Random Forest model using a grid-search. The grid-search is a great way to zero in on the best model hyperparameters that optimize the target metric - in this case, accuracy: 
param_grid = { 
    'n_estimators': [50, 100, 150],
    'max_depth' : [5, 15, 30],
    'min_samples_leaf' : [1, 5, 10],
    'criterion' : ['gini', 'entropy']
}
************************************************
rfc = RandomForestClassifier(random_state = 42)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
CV_rfc.fit(X_train, y_train)
************************************************
print('Optimal hyperparameters for max accuracy score:', CV_rfc.best_params_)
The above process took several hours to complete (grid searches can be computationally expensive) but provided the best hyperparameters which increased the model's performance from an accuracy score of ~82% to ~85%. 
As a last measure, I assessed the features that contributed the most predictive information: 
feature_importances_2 = pd.Series(trained_model_2.feature_importances_, index = X_train.columns).sort_values(ascending = False)
************************************************
plt.figure(figsize=(13,10))
sns.barplot(x = feature_importances_2, y = feature_importances_2.index, color = 'cornflowerblue')
plt.title('Feature Importance', size = 15)
plt.yticks(size = 13)
plt.xticks(size = 12)
plt.show()
Which produced the following graphic: 
Interestingly, the hour at which the crime occurred ended up being the most important feature used by the algorithm for tuning its parameters!


---

To conclude, an 85% accuracy score for a classification model is hardly good enough to use in any real-world application. That said, for a dataset with 8 classes, this score was surprisingly good.
Attempts to improve the model's performance should include the following:
More Features: Acquire additional data containing features such as school districts, mass transit station stops, socioeconomic data, et cetera. These additional features, along with many potential others, may contain predictive information that might contribute to better model performance.
More Training Data: For this project, we trained our model using 2018 and 2019 YTD crime data. Further attempts to improve the model should include fitting the parameters using data from multiple years. Doing so is costly in terms of processing power and time, but the improvements in accuracy may be worth the effort.
Additional Classifiers: Although we employed 5 algorithms adept at classification problems, more classifiers exist. I'd recommend testing additional, perhaps less popular, classifying algorithms to rule out Random Forests as our peak performer given these data.



---

I hope this post was both informative and interesting!
