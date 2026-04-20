from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess, split_data, train_model, evaluate_model

df = load_data()
X, y = preprocess(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = KernelRidge(alpha=1.0, kernel='rbf')
model = train_model(model, X_train, y_train)
mse = evaluate_model(model, X_test, y_test)

print(f"KernelRidge - Average MSE on Test Set: {mse:.4f}")