## 21.1 Lesson Plan - Introduction to Machine Learning

### Please take the End-of-Course Instructional Staff Survey

Trilogy, as a company, values transparency and data-driven change quite highly. As we grow, we know there will be areas that need improvement. It’s hard for us to know what these areas are unless we’re asking questions. Your candid input truly matters to us, as you are key members of the Trilogy team. In addition to the individual feedback at the end of lesson plans
we would appreciate your feedback at following link:
[https://docs.google.com/forms/d/e/1FAIpQLSdWXdBydy047_Ys1wm6D5iJY_J-0Mo0BqCjfGc4Er2Bz9jo5g/viewform](https://docs.google.com/forms/d/e/1FAIpQLSdWXdBydy047_Ys1wm6D5iJY_J-0Mo0BqCjfGc4Er2Bz9jo5g/viewform)

### Overview

Today's lesson plan introduces students to classical machine learning algorithms in the context of [sklearn](http://scikit-learn.org/stable/), covering data preprocessing and common machine learning algorithms.

### Class Objectives

* Students will understand how to calculate and apply regression analysis to datasets.

* Students will understand the difference between linear and non-linear data.

* Students will understand how to quantify and validate linear models.

* Students will understand how to apply scaling and normalization as part of the data preprocessing step in machine learning.

### Instructor Notes

* Today's class introduces students to machine learning through the Scikit-Learn library. Scikit-Learn provides a consistent interface for all of their models that students should find encouraging.

* Some of the material today may feel repetitive (regression analysis), but it is important that students have a solid foundation in regression analysis both as a primary skill and as a building block for other machine learning algorithms.

* It is important to stress that the concept of creating a model, fitting (training) that model to the data, and then using it to make predictions has become the standard paradigm used in many modern machine learning libraries. This common interface makes it easy to experiment with new algorithms and libraries when exploring machine learning solutions. Students will learn that there is no single "right algorithm" to use for any particular dataset or problem and that experimentation and validation is often preferred. Students will learn to quantify and validate the performance of many models on a dataset to determine which model may be best suited for their needs.

* Have your TAs refer to the [Time Tracker](TimeTracker.xlsx) to stay on track.

### Sample Class Video (Highly Recommended)

* To view an example class lecture visit (Note video may not reflect latest lesson plan): [Class Video](https://codingbootcamp.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9644eee9-4673-4107-bfc7-a89100f924ba)

- - -

### 01. Instructor Do: Introduction to Machine Learning (0:15)

* Step through the [Intro_to_ML slideshow](Slide-Shows/Intro_to_ML.pptx) slides 1-26.

### 02. Instructor Do: Univariate Linear Regression (0:10)

* **Files**:
  [Ins_Univariate_Linear_Regression_Sklearn.ipynb](Activities/01-Ins_Univariate_Linear_Regression_Sklearn/Solved/Ins_Univariate_Linear_Regression_Sklearn.ipynb)

* Open the Jupyter Notebook file and start the Rise slideshow.

* Explain that we will be discussing one of the fundamental algorithms in Machine Learning, Linear Regression.

  * Linear Regression has strong roots in statistics and is often used as a building block for other Machine Learning algorithms such as Neural Networks and deep learning.

  * Linear Regression is fast! If the problem can be solved with Linear Regression, that it is often more efficient and economical to use LR over a more complex model such as Deep Learning.

  * Many data scientists first start with a linear regression model and only move onto a more complex algorithm if their data proves to be truly non-linear.

* Ask the students for a definition of linear data.

  * Explain that linear data will visually represent a line.  Data that is perfectly linear will have a constant rate of change.

  * Explain that we will use a Sklearn function called `make_regression` to generate some test data.

    * Walk through the parameter list for `make_regression` and explain that we are defining 20 samples (rows) with 1 feature (column) and some noise and bias.

  * Use Matplotlib to plot the data and show the linear trend.

    * Explain that as X increases, y increases by a rate that is roughly constant.

      ![trend.png](Images/trend.png)

  * Explain that linear data can also have a negative trend; as the independent value (x) increases, the dependent value (y) decreases.

* Show the formula for Univariate Linear Regression and explain that this is just finding a line that best fits the data.

  ![linear_regression.jpg](Images/linear_regression.jpg)

* Use the home price example to illustrate the process of acquiring new data (a new house on the market) and using linear regression to predict the home price.

  ![predict_prices_3.png](Images/predict_prices_3.png)

* Briefly discuss non-linear data using the examples provided in the slideshow.

  ![nonlinear.png](Images/nonlinear.png)

* Model - Fit - Predict

  * Explain that many popular machine learning libraries follow a model - fit - predict pattern. Walk the students through an example of this using Linear Regression in sklearn.

    ![sklearn_api.png](Images/sklearn_api.png)

  * Explain that we are going to import `LinearRegression` from Sklearn and instantiate a model from it.

  * Explain that once we have a model instantiated, we need to fit the model to the data. This is the training process.

    * Explain that the goal with training is to find the slope and intercept that best represents the data (fitting a line to the data).

  * Show the slope and intercept for the model using `model.coef_` for the slope and `model.intercept_` for the y-axis intercept.

    ![coeff.png](Images/coeff.png)

  * Explain that we can now use the line to make predictions for new inputs. We now have a model that can take any value of X and calculate a value of y that follows the trend of the original data.

  * Compare the first prediction to the original output value. These two values should be very close to each other because the model represents the trend of the original data.

  * Use the min and max values for X to make predictions for y. Compare that to the original data to show that they should be fairly close in value.

  * Plot the original data vs the predicted min and max values. This will visually show how well the model fits the original data.

    ![line_fit.png](Images/line_fit.png)

### 03. Students Do: Stu_Univariate_Linear_Regression (0:15)

* In this activity, students calculate a regression line using a dataset of lsd drug concentrations vs. math scores.

* **File**: [Stu_LSD.ipynb](Activities/02-Stu_LSD/Unsolved/Stu_LSD.ipynb)

* **Instructions:** [README.md](Activities/02-Stu_LSD/README.md)

  * Start by creating a scatter plot of the data to visually see if any linear trend exists.

  * Next, use sklearn's linear regression model and fit the model to the data.

    * Print the weight coefficients and the y-axis intercept for the trained model.

  * Calculate the `y_min` and `y_max` values using `model.predict`

  * Plot the model fit line using `[x_min[0], x_max[0]], [y_min[0], y_max[0]]`

### 04. Everyone Do: Review Activity (0:10)

* Reassure students that it's okay if this was difficult. The Sklearn and TensorFlow libraries share a common api, so once you master the `Model-Fit-Predict` steps, it is easy to switch to other Machine Learning Models later one. They will get plenty of practice with this today!

* Open up [Stu_LSD.ipynb](Activities/02-Stu_LSD/Solved/Stu_LSD.ipynb).

* During the review, highlight the following points:

  * Show how to assign the data and target to variables `X` and `y`.

    * Explain that it is not necessary to use `X` and  `y`, but it does provide a consistent set of variable names to use with our models.

    * Explain that we have to call `reshape(-1, 1)` to format the array for sklearn. This is only necessary for 1-dimensional array.

    ![reshape.png](Images/reshape.png)

  * Plot **X** and **y** to show the linear trend in the data.

    * Point out that it is ok to have a negative slope in this case. The data still follows a linear trend.

      ![negative_trend.png](Images/negative_trend.png)

  * Show how to instantiate and fit a model to the data.

  * Print the slope and intercept values and remind students that we are simply defining the equation for the line.

  * Plot the line and the original data to show visually how well the line fits the model.

  * Ask students what it might mean if the line did not appear to match the data well. Explain that it may indicate that the model was not a good fit, or that there were errors somewhere in the code.

    ![lsd_regression_line.png](Images/lsd_regression_line.png)

### 05. Instructor Do: Quantifying Regression (0:10)

* In this activity, two popular metrics to quantify their machine learning models are shown. The importance of validation by splitting data into training and testing sets is also covered.

* Open [Intro_to_ML.pptx](Slide-Shows/Intro_to_ML.pptx) to go over slides 27-32 and open [Ins_Quantifying_Regression.ipynb](Activities/03-Ins_Quantifying_Regression/Solved/Ins_Quantifying_Regression.ipynb) in Jupyter Notebook.

* Quantification

  * Go over slide 28 while explaining that more than visual confirmation of a model is necessary to judge its strength. The model must be quantified. Two common quantification scores are **Mean Squared Error (MSE)** and **R Squared (R2)**.

  * Sklearn provides functions to calculate these metrics.

  * Switch to [Ins_Quantifying_Regression.ipynb](Activities/03-Ins_Quantifying_Regression/Solved/Ins_Quantifying_Regression.ipynb) to show how to use `sklearn.metrics` to calculate the **MSE** and **R2** scores.

  * Point out that a "good" MSE score will be close to zero while a "good" R2 Score will be close to 1.

  * Explain that R2 is the default score for a majority of Sklearn models. It can be calculated directly from the model using `model.score`.

* Validation

  * Switch back to [Intro_to_ML.pptx](Slide-Shows/Intro_to_ML.pptx) and go over slides 29-32.

  * In order to understand how the model performs on new data, the data is split into training and testing datasets. The model is fit (trained) using training data, and scored/validated using the testing data.  This gives an unbiased measure of model effectiveness.

  * This train/test splitting is so common that Sklearn provides a mechanism for doing this. Show students how to use the `train_test_split` function to split the data into training and testing data using [Ins_Quantifying_Regression.ipynb](Activities/03-Ins_Quantifying_Regression/Solved/Ins_Quantifying_Regression.ipynb).

### 06. Students Do: Brains! (0:15)

* In this activity, students calculate a regression line to predict head size vs. brain weight.

* **File**: [Stu_Brains.ipynb](Activities/04-Stu_Brains/Unsolved/Stu_Brains.ipynb)

* **Instructions:** [README.md](Activities/04-Stu_Brains/README.md)

  * Start by creating a scatter plot of the data to visually see if any linear trend exists.

  * Split the data into training and testing using sklearn's `train_test_split` function.

  * Next, use sklearn's linear regression model and fit the model to the training data.

  * Use the test data to make new predictions. Calculate the MSE and R2 score for those predictions.

  * Use `model.score` to calculate the the R2 score for the test data.

### 07. Everyone Do: Review Activity (0:10)

* Remind students that the data must be reshaped because sklearn expects the data in a particular format.

* Ask the students why the MSE score is so large. Explain that this is because MSE is not upper bounded. Optionally, slack out the formula for [MSE](https://en.wikipedia.org/wiki/Mean_squared_error).

* Highlight is that the model should always perform better on the training set than the testing set. This because the model was trained on the training data and not on the testing data. Intuitively, the model should perform better on data that it has seen before versus data it has not seen.

* Note that `r2_score` and `model.score` produce the same R2 score.

- - -

### 08. BREAK (0:15)

- - -

### 09. Instructor Do: Multiple Linear Regression (0:10)

* In this activity, we discuss multiple (multi-feature) linear regression.

  * Explain that multiple linear regression is linear regression using multiple input features. Use the house price example as an analogy. With univariate linear regression, the trend might predict the price of a home dependent on one variable: square feet.  Multiple linear regression allows multiple inputs such as the number of bedrooms, number of bathrooms, as well as square feet.

      ![multiple_regression.png](Images/multiple_regression.png)

  * Explain that with multiple linear regression, it becomes hard to visualize the linear trends in the data. We need to rely on our regression model to correctly fit a line. Sklearn uses the Ordinary Least Squares method for fitting the line. Luckily for us, the api to the linear model is exactly the same as before! We simply fit our data to our n-dimensional X array.

      ![3dplot.png](Images/3dplot.png)

* Residuals

  * Explain that with multi-dimensional data, we need a new way to visualize our model performance. In this example, we use a residual plot to check our prediction performance. Residuals are the difference between the true values of y and the predicted values of y.

      ![residuals.png](Images/residuals.png)

### 10. Students Do: Beer Foam (0:15)

* **File**: [Stu_Beer_Foam.ipynb](Activities/06-Stu_Beer_Foam/Unsolved/Stu_Beer_Foam.ipynb)

### 11. Everyone Do: Review Activity (0:10)

* Explain that we are now using 2 features for our X data, `foam` and `beer`. Using more than one feature (independent variable) is considered multiple regression.

* Show that our api is the same. That is, we still use the model, fit, predict interface with sklearn. Only the dimensionality of the data has changed. Point out that we do not have to `reshape` our X data because it is already in the format that sklearn expects. Only 1 dimensional input vectors have to be reshaped.

* Explain that we will often see a higher r2 score using multiple regression over simple (1 independent variable) regression. This is because we are using more data to make our predictions.

* Show the residual plot for this model using both training and testing data. We do have outliers in this plot which may indicate that our model would not perform as expected. It's hard to say without testing with more data points.

    ![residuals_beer_foam.png](Images/residuals_beer_foam.png)

### 12. Instructor Do: Data Preprocessing (0:15)

* This activity discusses several important data pre-processing techniques necessary for many machine learning algorithms.

  * The first big concept is how to convert text and categorical data to numeric features. Most algorithms require label, one-hot, or binary encoding for the data.

      ![categorical_data.png](Images/categorical_data.png)

    * Label encoding is one approach where each category is encoded as an integer value. However, certain machine learning algorithms are sensitive to integer encoding.

      ![label_encoding.png](Images/label_encoding.png)

    * The Pandas `get_dummies` function should be sufficient for most situations and can be applied to the entire dataframe at once.

    * Note that the example dataframe contains what appears to be numerical values in the `age` column, but these are actually text values that represent an age range. These need to be converted to binary numerical values.

      ![dummy_encoding.png](Images/dummy_encoding.png)

  * The second big concept is scaling and normalization. The primary motivation for scaling is to shift all features to the same numeric scale so that large numerical values do not bias the error calculations during the training cycles.

    * It is recommended to split your data into training and testing data before fitting the scaling models. This is so that you do not bias your results by using the testing data to calculate the scale. The test data should be completely independent of the training step.

    * The Sklearn developers recommend using standard scaler when you do not know anything about your data.

      ![standard_scaler.png](Images/standard_scaler.png)

    * Another common scaling technique is MinMax scaling where values are scaled from 0 to 1.

      ![minmax_scaler.png](Images/minmax_scaler.png)

    * The only time that you may not want to scale is if the magnitudes of your input features has significance that needs to be preserved (i.e. Pixel values in the MNIST handwriting recognition dataset).

    * Note that scaling and normalization will often result in much more reasonable MSE values.

### 13. Students Do: Respiratory Disease (0:20)

* **File**: [Stu_Respiratory_Disease.ipynb](Activities/08-Stu_Respiratory_Disease/Unsolved/Stu_Respiratory_Disease.ipynb)

### 14. Everyone Do: Review Activity (0:10)

* Explain that our dataset has categorical values for the columns `sex` and `smoker`. We need to use the pandas `get_dummies` function to convert these to binary values.

  * Remind students that `get_dummies` will automatically create new columns for each category.

* Remind students that we need to fit our scaler model to the training data only. We do not use the testing data because we do not want to bias our scaler with the testing data.

  * Show that we can then apply the scaler model to our training and testing data.

* Explain that though we didn't explicitly cover `Lasso`, `Ridge`, and `ElasticNet`, these algorithms follow the same `model->fit->predict` pattern consistent with linear regression.

    ![linear-models.png](Images/linear-models.png)

* Point out that all four of the models had similar performance for this particular dataset, but that may not always be the case. It's very common in machine learning to test several models on your dataset to see which model has the best performance. In this case, there were no significant advantages to using more complicated algorithms, so linear regression is probably still the best choice.

- - -

### LessonPlan & Slideshow Instructor Feedback

* Please click the link which best represents your overall feeling regarding today's class. It will link you to a form which allows you to submit additional (optional) feedback.

* [:heart_eyes: Great](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=21.1&lp_useful=great)

* [:grinning: Like](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=21.1&lp_useful=like)

* [:neutral_face: Neutral](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=21.1&lp_useful=neutral)

* [:confounded: Dislike](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=21.1&lp_useful=dislike)

* [:triumph: Not Great](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=21.1&lp_useful=not%great)

- - -

### Copyright

Trilogy Education Services © 2017. All Rights Reserved.
