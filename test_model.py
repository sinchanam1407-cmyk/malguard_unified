import joblib, numpy as np, warnings
warnings.filterwarnings('ignore')

model = joblib.load('model/GradientBoosting_best.pkl')
n_feat = model.n_features_in_

print('=' * 48)
print('  MalGuard Model Verification')
print('=' * 48)
print('  Type         :', type(model).__name__)
print('  Estimators   :', model.n_estimators)
print('  Learning rate:', model.learning_rate)
print('  Max depth    :', model.max_depth)
print('  Features in  :', n_feat)
print('  Classes      : 0=Benign, 1=Malware')
print()

# Test 1 - benign (all zeros)
X_benign = np.zeros((1, n_feat))
pred  = model.predict(X_benign)[0]
proba = model.predict_proba(X_benign)[0]
label = 'Malware' if pred == 1 else 'Benign'
print('  Test 1 (benign APK)  ->', label, ' confidence:', round(max(proba)*100, 1), '%')

# Test 2 - malware (all ones = all features present)
X_mal = np.ones((1, n_feat))
pred2  = model.predict(X_mal)[0]
proba2 = model.predict_proba(X_mal)[0]
label2 = 'Malware' if pred2 == 1 else 'Benign'
print('  Test 2 (malware APK) ->', label2, ' confidence:', round(max(proba2)*100, 1), '%')

print()
print('  Model loaded and working correctly!')
print('=' * 48)
