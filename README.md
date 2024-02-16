# LFP-mediated-aversion-prediction

**Introduction**: The kappa-opioid receptor (KOR) system plays a key role in modulating stress responses, aversive behaviors, and dysphoric states. In our study, we developed a model capable of predicting the intensity of aversive reactions by analyzing the brain's bioelectrical activity.

**Materials and Methods**: Rats underwent stereotaxic surgery to implant electrodes into the amygdala, ventral tegmental area, prefrontal cortex, hippocampus, and nucleus accumbens. The rats were subjected to conditioned place preference test to measure aversion to U-50488. After that, local field potential (LFP) recordings were made. LFP data were processed and analyzed using spectral and coherence analysis methods. A stepwise multiple linear regression was employed to identify the LFP features most significantly correlated with aversion to kappa-opioid agonist U-50488.
![image001](https://github.com/neuroeeg/LFP-mediated-aversion-prediction/assets/67147000/bb58ff46-d719-414d-9885-e065cd21c5b7)
**Results**: The administration of U-50488 resulted in significant changes in LFP signals across multiple brain regions. These changes were particularly notable in the theta, gamma, and delta bands of brain waves (p<0.05). Theta and gamma activities were especially sensitive to the effects of kappa-opioid agonist U-50488. Connectivity calculations revealed shifts in coherence between brain regions, particularly highlighting the amygdala's involvement. While changes were also observed in the ventral tegmental area, prefrontal cortex, hippocampus, and nucleus accumbens (p<0.05), they contributed less to aversion. Using the stepwise multiple linear regression method, we established a final model with the 3 most significant variables: (1) coherence between the amygdala and medial prefrontal cortex, (2) coherence between the amygdala and hippocampus, and (3) theta power in the amygdala.

As can be seen in Fig. 2, the predicted values for aversion were in good agreement with the actual values. Here we applied 5-fold cross-validation to validate our results. Our findings reveal a positive significant correlation of r = 0.72±0.02, between the actual and the predicted values, with R-squared = 0.18±0.08, MAE = 11.69±0.08 with p = 0.0099 (Figure 2 shows the same statistical measures for the whole dataset).
![image003](https://github.com/neuroeeg/LFP-mediated-aversion-prediction/assets/67147000/289e65fd-99a5-41d3-b362-6daecfb97c6e)
Figure 2. (A) Linear correlation plot between actual and predicted values of aversion for the whole dataset. Predictors selected for the multiple linear regression model: Amy-mPFC coherence, Amy-Hipp coherence and LFP theta power in Amy (all p<0.05). The right-hand margin histogram provides a depiction of the distribution of predicted values, while the top margin histogram illustrates the distribution of actual values. (B) Represents the graph of residuals vs. predicted values where the prediction made by the model is on the x-axis and the accuracy of that prediction is on the y-axis. The residuals are symmetrically distributed around zero. Histograms in the upper and right margins represent the distributions of predicted values and residuals.

**Conclusion**: Overall, the data provided insights into how electrical neural activity mediates aversion in response to KOR activation. The results showed that the severity of aversion can be reasonably predicted (r = 0.72±0.02, p = 0.0099) using LFP band power and functional connectivity data.

# Multiple linear regression model construction and validation

File: 5-fold_cross-validation.py

Key steps and components:

1. Library Imports: Essential Python libraries for data manipulation, statistical modeling, and mathematical operations are imported, including pandas for data frames, NumPy for numerical operations, sklearn for machine learning models and metrics, and scipy for scientific computing tasks like correlation calculations.
2. Data Loading: The dataset is loaded from a CSV file into a pandas DataFrame. This dataset contains the independent variables ('Amy-mPFc', 'Amy-Hipp', 'tetha power') and the dependent variable ('aversion').
3. Model Preparation: A linear regression model is prepared for fitting the data. Additionally, parameters for cross-validation (using KFold) and permutation testing are set up.
4. Cross-Validation and Metrics Calculation: The dataset is divided into training and testing sets multiple times using K-fold cross-validation. For each split, the model is trained on the training set and evaluated on the testing set. Metrics such as Mean Absolute Error (MAE), R-squared, Mean Squared Error (MSE), and Pearson correlation coefficients are calculated to assess model performance.
5. Permutation Testing: The dependent variable values ('aversion') are shuffled (permuted) multiple times, and the model is retrained and evaluated with these permuted labels to create a distribution of performance metrics under the null hypothesis (no relationship between predictors and outcome). This step helps assess the significance of the observed relationship between predictors and the outcome in the original data.
6. Final Evaluation and Output: The actual MAE for the original dataset is calculated and compared against the distribution of MAEs from the permutation tests to compute a p-value, indicating the significance of the findings. The mean values of R-squared, MSE, and the correlation coefficient are also calculated and reported to summarize model performance.
7. Results Reporting: The script prints the actual MAE of the model when applied to the unpermuted data, the mean of the permuted MAE values, a p-value for testing the hypothesis of no association, and average values of R-squared, MSE, and the Pearson correlation coefficient.

This approach, combining cross-validation for robust model evaluation and permutation testing for significance assessment, provides a comprehensive analysis of the predictive power and significance of the relationship between the selected predictors and the dependent variable in the context of multiple linear regression.

# Citation

Kalitin KY, Spasov AA, Mukha OY (2023) Aversion-related effects of kappa-opioid agonist U-50488 on neural activity and functional connectivity between amygdala, ventral tegmental area, prefrontal cortex, hippocampus, and nucleus accumbens. Research Results in Pharmacology 9(4): 21–29. https://doi.org/10.18413/rrpharmacology.9.10051

