# UC Berkeley | Professional Certificate in Machine Learning and Artificial Intelligence
## Capstone Project 2 (Module 24)

**Author**: 

Christian Fajardo
___

##### Note: Due to inaccessibility of data sources originally defined in Module 17, I needed to identify and select an alternative that is closely related to the manufacturing industry.

---

### Executive summary
#### Rationale

Why should anyone care about this question?

Preventive maintenance is critical in modern manufacturing to minimize unplanned downtime, reduce costs, and extend equipment lifespan. Using predictive machine learning (ML), manufacturers can anticipate failures before they occur by analyzing patterns in real-time operational data. 

This proactive approach enables timely interventions that avoid costly breakdowns and production delays. ML-driven maintenance also improves resource allocation by focusing efforts where they are needed most, rather than relying on fixed schedules. 

As a result, companies can significantly increase operational efficiency and asset reliability. Investing in predictive maintenance using ML is a strategic move that drives long-term savings and strengthens competitive advantage.
___

### Research Question
Will a manufacturing machine or product experience a failure that could disrupt production operations?

___

### Data Sources
Data source: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

- `UID`: unique identifier ranging from 1 to 10000
- `productID`: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
- `air temperature [K]`: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- `process temperature [K]`: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- `rotational speed [rpm]`: calculated from powepower of 2860 W, overlaid with a normally distributed noise
- `torque [Nm]`: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
- `tool wear [min]`: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a 'machine failure' label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.

Targets
- `Target` : Failure or Not
- `Failure Type` : Type of Failure

___

### Methodology
What methods are you using to answer the question?

**Feature Engineering**

For this dataset I have to do the following transformations:
- Dropping irrelevant column (`ProductId`)
- Rename columns to shorter but more meaningful names
- Use OneHotEncoder to extract numerical values from string-type categorical features
- Since this dataset is imbalanced, I needed to use `SMOTE` provided by the `imblearn` Python library


#### Model Selection

For this dataset, my methodology will include training supervised machine learning models such as K-Nearest Neighbors (KNN), Decision Trees, Logistic Regression, Support Vector Machines (SVM) to predict potential machine failures.

5-Fold Cross Validation is used to split the data into multiple training and testing subsets, ensuring that every sample is used for validation at least once. 

This approach reduces overfitting and provides a more robust estimate of the model's performance on unseen data.

Finally, I will use Deep Learning (Tensorflow/Keras) as well to perform multiclass classification on `Target` and `Failure Type`.

Here is the model summary:
- **Supervised Machine Learning Models**:
  - K-Nearest Neighbors (KNN):
    - Based on similarity among observations.
    - Requires careful tuning of the number of neighbors and distance metrics.
    
  - Decision Trees:
    - Offers clear decision rules and interpretability.
    - Can capture non-linear relationships between sensor readings and failures.
    
  - Logistic Regression:
    - Suitable for binary or multiclass classification with probabilistic outputs.
    - Assumes a linear relationship which can simplify the analysis.
    
  - Support Vector Machines (SVM):
    - Effective in high-dimensional spaces.
    - Requires kernel selection and parameter tuning to handle non-linear decision boundaries.

- **Deep Learning with Dense Layers:**
    - Utilizes a SoftMax layer for multiclass classification on both `Target` and `Failure Type`.
    - Potential to capture complex non-linear relationships and feature interactions.

- **Model Comparison:**
  - Compare traditional models with the CNN approach using evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
  - Consider aspects like model interpretability, training time, and robustness to overfitting.

___

### Results
The initial dataset intended for this research was not feasible to acquire due to access limitations. As a result, I pivoted to an alternative dataset that remains highly relevant to the manufacturing industry, aligning with my professional domain. 

The new dataset, sourced from Kaggle, is titled "Machine Predictive Maintenance Classification" and focuses on forecasting equipment failures using sensor data. It contains 10,000 rows and 14 columns, offering a rich set of features including temperature, rotational speed, torque, and tool wear.

These variables are directly applicable to real-world manufacturing operations, making the dataset suitable for predictive maintenance modeling. 
This shift ensures that the research remains grounded in practical industry applications while maintaining analytical rigor.

Data shows the correlation table shows a strong positive correlation between `Air Temperature` and `Process Temperature` (r = `0.88`), indicating they increase together. 

Rotational speed `RPM` and `Torque` have a strong negative correlation (r = -0.88), suggesting that higher speed is associated with lower torque. `Failure` has the highest positive correlation with `Torque` (r = `0.19`) and tool wear (r = `0.11`), though these correlations are still relatively weak. 

The rest of the variables show low direct correlation with failure.


### Model Performance
During the intial run, the models performed fairly well but it still have room for improvement:

#### Accuracy & Confusion Matrix

| **Model**                         | **Accuracy** | **Confusion Matrix**           |
|:----------------------------------|:------------:|:------------------------------:|
| **K-Nearest Neighbor (K=5)**      | 0.9790       | [[1935, 4], [38, 23]]          |
| **Decision Tree**                 | 0.9795       | [[1932, 7], [34, 27]]          |
| **Logistic Regression**           | 0.9735       | [[1931, 8], [45, 16]]          |
| **SVM (Linear Kernel)**           | 0.9770       | [[1936, 3], [43, 18]]          |



#### Deep Learning Metrics

Model Architecture

| Layer (type)     | Output Shape | Param #   |
|------------------|--------------|-----------|
| dense_15 (Dense) | (None, 100)  |       900 |
| dense_16 (Dense) | (None, 100)  |    10,100 |
| dense_17 (Dense) | (None, 100)  |    10,100 |
| dense_18 (Dense) | (None, 1)    |       101 |


- Total params: 42,404 (165.64 KB)
- Trainable params: 21,201 (82.82 KB)
- Non-trainable params: 0 (0.00 B)
- Optimizer params: 21,203 (82.83 KB)


&nbsp;  

**Model Metrics** 

Training Accuracy:  `0.9748769998550415`

Testing Accuracy:  `0.953499972820282`

&nbsp;  
**Confusion Matrix** 
|                      | **Predicted Negative** | **Predicted Positive** |
|----------------------|------------------------|------------------------|
| **Actual Negative**  | TN = 1865              | FP = 74                |
| **Actual Positive**  | FN = 12                | TP = 49                |

&nbsp;

**Overall Model Performance**
| Metric    | Value   |
|-----------|---------|
| Accuracy  | 0.957   |
| Precision | 0.398   |
| Recall    | 0.803   |

&nbsp;


# Conclusion

Deep learning using TensorFlow outperformed traditional models by demonstrating a robust balance between high overall accuracy and effective identification of equipment failures. 

Its training accuracy of 97.62% and testing accuracy of 95.70% indicate that it generalizes well to unseen data. The confusion matrix—showing only 12 false negatives and 74 false positives—highlights its strong ability to correctly detect failures while minimizing missed alerts. Fewer false negatives are critical for proactive maintenance, ensuring that potential issues are caught early before leading to costly downtime. 

This predictive strength allows maintenance to be scheduled optimally, significantly reducing unexpected equipment failures and operational interruptions. 

Overall, deep learning is the best model for predicting and scheduling equipment maintenance, ultimately saving the manufacturing company money through improved efficiency and reliability.


### Next Steps
- Deploy the model
- Gather more data if possible
- Apply more fine-tuning techniques, Random Forest, various hyper-parameter trials
- Perform continous model training and deployment cycles
- Research on identifying the optimal, best hyper-parameters for Neural Networks (e.g. _Deep Learning with Python_ by Francois Chollet)

___

### Outline of project

- `README.md`: Documentation.
- `data/`: Contains the cleaned dataset used for analysis.
- `images/`: Contains plot images.
- `capstone_proj_2_final_report.ipynb`: Jupyter notebook
___

### Contact and Further Information

Christian Fajardo

Email: ulchrisf@gmail.com

[LinkedIn](linkedin.com/in/christian-fajardo-01632823)