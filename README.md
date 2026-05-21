# UCU-Spring-2026-Prediction-of-the-outcome-of-Tennis-Matches

Create a coursework that compares the pairwise and pointwise approaches for predicting outcomes of tennis matches. Do research on state-of-the-art feature-engineering-based LTR algorithms that perform well on small datasets, both for the pointwise and the pair-wise approaches. The datasets will be small because top-level professional tennis players generally play between 50 and 70 matches per year. Make sure to include gradient boosting approaches and linear models, such as SVMs.

1. Data gathering
   1. Identify potentially useful features and their availability.
   2 Gather weekly rankings of the top 50-100 tennis players from the WTA and ATP websites for 2025 and 2024
   3. Gather the results of every tournament where selected players participated, along with all the players who participated in that tournament, and the statistics useful for the feature engineering.
   4. Gather weekly ranking for missing players.
2. Experiments
   1. Identify possible formulations of the task for point-wise and list-wise approaches.
   2. Make a rationale for the chosen data split strategy.
   3. Create the simplest model based on the recent Ranking to predict the outcome.
   4. Perform the feature engineering.
   5. Train new models
   6. Perform a simulation of season 2025 based on a rolling window strategy (update the features for the next matches based on the previous ones)
3. Key point regarding reporting
   1. What is the key difference in terms of tasks that can be done by pairwise and pointwise approaches?
   2. Perform the feature analysis based on shap-values.
   3. What was the role of hyperparameter tuning, and which methods worked the best?
   4. Which models have demonstrated the best performance, and which aspect had the greatest impact on the results?