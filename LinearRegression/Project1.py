import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# DATA VERILERINI OKUTUYORUZ.
train_data = pd.read_csv('trainDATA.csv')
test_data = pd.read_csv('testDATA.csv')
train_data.drop([1312], inplace = True)
test_data.columns=['name','year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner','selling_price']

# Categoric Variableları encode eden fonksiyon.
def encode_categorical_data(data, column_name, encoding_map):
    data[column_name] = data[column_name].map(encoding_map)

# Categoric Variable'lar için verilen değerler.
fuel_encoding = {'Diesel': 1, 'Petrol': 2}
seller_type_encoding = {'Individual': 1, 'Dealer': 2, 'TrustmarkDealer': 3}
transmission_encoding = {'Manual': 1, 'Automatic': 2}
owner_encoding = {'FirstOwner': 1, 'SecondOwner': 2, 'ThirdOwner': 3, 'Fourth&AboveOwner': 4}

# Categoric Variableları tek tek encode eder.
encode_categorical_data(train_data, 'fuel', fuel_encoding)
encode_categorical_data(train_data, 'seller_type', seller_type_encoding)
encode_categorical_data(train_data, 'transmission', transmission_encoding)
encode_categorical_data(train_data, 'owner', owner_encoding)

encode_categorical_data(test_data, 'fuel', fuel_encoding)
encode_categorical_data(test_data, 'seller_type', seller_type_encoding)
encode_categorical_data(test_data, 'transmission', transmission_encoding)
encode_categorical_data(test_data, 'owner', owner_encoding)

# Train verileri.
X_train = train_data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']].values
y_train = train_data['selling_price'].values
# Test verileri.
X_test = test_data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']].values
y_test = test_data['selling_price'].values

max_y = y_train.max()
min_y =  y_train.min()

# Column'lara feature scaling uygular. (X -Xmin / (Xmax - Xmin))
def min_max_scaling(data):
    min_vals = np.min(data,axis=0)
    max_vals = np.max(data,axis=0)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Feature scaling uygulanmış veriler. (Kullandığım Scaling Formülü : (X -Xmin / (Xmax - Xmin))
X_train_scaled = min_max_scaling(X_train)
X_test_scaled = min_max_scaling(X_test)
y_train_scaled = min_max_scaling(y_train)
y_test_scaled = min_max_scaling(y_test)

# X0 = 1 columnu için ilk column'a 1'ler ekledim.
X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

#Cost hesaplayan fonksiyon.
def cost_func(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    error = predictions - y
    cost= (1 / (2 * m)) * np.sum(error ** 2)
    return cost

#Gradient Descent Uygulayan Fonksiyon.
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        predictions = np.dot(X, theta)
        error = predictions - y
        gradient = (1 / m) * np.dot(X.T,error)
        theta -= alpha * gradient
        cost = cost_func(X, y,theta)
        cost_history.append(cost)

    return theta, cost_history

# Gereken Parametreler
alpha = 0.01
num_iterations = 800
initial_theta = np.zeros(X_train_scaled.shape[1])
theta_initial_copy = initial_theta.copy()


# Train Data üzerinde uygulanan gradient descent ve cost hesaplamalarından elde edilen veriler.
final_theta, cost_history= gradient_descent(X_train_scaled, y_train_scaled, theta_initial_copy, alpha, num_iterations)


# İlk ve Son Theta parametre değeleri.
print(f'Theta Initial: {initial_theta}\nTheta Final:{final_theta}\n')

# Train Data üzerinden elde edilen ilk ve son cost değerleri.
print(f'Cost History:{cost_history}\n')
print(f'First Cost of Train Data : {cost_history[0]} \nLast Cost of Train Data : {cost_history[-1]}\n')

# Test verileri üzerinde tahmin.
y_predicted = np.dot(X_test_scaled,final_theta)
y_predicted_actual = y_predicted * (max_y - min_y)  + min_y # max_y=8150000 ,min_y=20000

# Selling price üzerindeki Test ve Tahmin verilerinin outputu.
for i in range (len(y_test)):
   print(f'{i+1}. test data : {y_test[i]} & prediction(reverse scaled) : {y_predicted_actual[i]}')


# Selling price üzerindeki scaled Test ve sclaed Tahmin verilerinin outputu.
#for  i in range (len(y_test)):
#    print(f'AND TO SEE AS SCALED VERSION => {i+1}. test data scaled : {y_test_scaled[i]} & prediction scaled  : {y_predicted[i]}')


# y_predicted ve y_test arasındaki error'e bakıyoruz.
y_test_cost_mse = (1 / (2 * len(initial_theta)))* np.sum((y_predicted - y_test_scaled) ** 2)
print(f'Error between predicted y values and actual y values on testData: {y_test_cost_mse}')


#Gradient Descent kullanılarak, Cost'un İterasyon'a bağlı değişim grafiği.
plt.plot(range(num_iterations), cost_history, label=f'Learning Rate: {alpha}')
plt.xlabel('İterasyon Sayısı')
plt.ylabel('Maliyet')
plt.title('Gradient Descent İterasyonlarına Göre Cost')
plt.legend()
plt.show()