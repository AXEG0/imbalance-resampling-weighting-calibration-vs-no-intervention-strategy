from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler

# Create the data
X, y = make_classification(n_samples=100000, n_features=4, random_state=42, weights=[0.95, 0.05])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
X_train_rus, y_train_rus = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

# Create a logistic regression model
model = LogisticRegression(random_state=42)
model_balanced = LogisticRegression(class_weight='balanced', random_state=42)
model_resampled = LogisticRegression(random_state=42)
calibrated_model = CalibratedClassifierCV(model_balanced)

# Fit the model to the training data
model.fit(X_train, y_train)
model_balanced.fit(X_train, y_train)
model_resampled.fit(X_train_rus, y_train_rus)
calibrated_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict_proba(X_test)
y_pred_balanced = model_balanced.predict_proba(X_test)
y_pred_resampled = model_resampled.predict_proba(X_test)
y_pred_calibrated = calibrated_model.predict_proba(X_test)

# Get probabilities
y_pred = y_pred[:, 1]
y_pred_balanced = y_pred_balanced[:, 1]
y_pred_resampled = y_pred_resampled[:, 1]
y_pred_calibrated = y_pred_calibrated[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred)
auc_balanced = roc_auc_score(y_test, y_pred_balanced)
auc_resampled = roc_auc_score(y_test, y_pred_resampled)
auc_calibrated = roc_auc_score(y_test, y_pred_calibrated)

# Brier score
brier = brier_score_loss(y_test, y_pred)
brier_balanced = brier_score_loss(y_test, y_pred_balanced)
brier_resampled = brier_score_loss(y_test, y_pred_resampled)
brier_calibrated = brier_score_loss(y_test, y_pred_calibrated)

print('AUC: {:.2f}'.format(auc))
print('AUC Balanced: {:.2f}'.format(auc_balanced))
print('AUC Resampled: {:.2f}'.format(auc_resampled))
print('AUC Calibrated: {:.2f}'.format(auc_calibrated))
print()
print('Brier: {:.2f}'.format(brier))
print('Brier Balanced: {:.2f}'.format(brier_balanced))
print('Brier Resampled: {:.2f}'.format(brier_resampled))
print('Brier Calibrated: {:.2f}'.format(brier_calibrated))

# Output:

# AUC: 0.90
# AUC Balanced: 0.90
# AUC Resampled: 0.90
# AUC Calibrated: 0.90

# Brier: 0.03
# Brier Balanced: 0.11
# Brier Resampled: 0.11
# Brier Calibrated: 0.03
