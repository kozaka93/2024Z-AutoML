import pandas as pd

from numerai_automl.scorer.scorer import Scorer

X = pd.read_csv("return_data.csv")

# X = X.rename(columns={'predictions_meta_weighted': 'predictions_model_meta_weighted'})
# X = X.rename(columns={'predictions_meta_lgbm': "predictions_model_meta_lgbm"})
# X = X.rename(columns={'predictions_omega': "predictions_model_omega"})



# X.to_csv("return_data.csv")




scorer = Scorer()
# scores = scorer.compute_scores(X[["predictions_omega", "target", "predictions_meta_weighted", "predictions_meta_lgbm", 'era']], "target")
scores = scorer.compute_scores(X, "target")

print(scores)

scores.to_csv("return_data_for_scoring.csv")



print(scores)