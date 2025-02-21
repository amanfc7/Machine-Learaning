
# TODO: simple setup for utilizing another AUTO ML (pycaret) implementation. might just be done directly in comparison, maybe

# Classification OOP API Example

# loading sample dataset
from pycaret.datasets import get_data
data = get_data('juice')

# init setup
from pycaret.classification import ClassificationExperiment
s = ClassificationExperiment()
s.setup(data, target = 'Purchase', session_id = 123)

# model training and selection
best = s.compare_models()

# evaluate trained model
s.evaluate_model(best)

# predict on hold-out/test set
pred_holdout = s.predict_model(best)

# predict on new data
new_data = data.copy().drop('Purchase', axis = 1)
predictions = s.predict_model(best, data = new_data)

# save model
s.save_model(best, 'best_pipeline')