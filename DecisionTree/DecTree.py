import pandas as pd
import numpy as np

# Veri setini okuma işlemi
col_names = ["Price", "MaintPrice", "NoofDoors", "Persons", "Lug_size", "Safety", "Car Acceptibility"]
train_data = pd.read_excel(r"trainDATA.xlsx", names=col_names)
test_data = pd.read_excel(r"testDATA.xlsx", names=col_names)

X_train = train_data.drop("Car Acceptibility", axis=1)
y_train = train_data["Car Acceptibility"]
X_test = test_data.drop("Car Acceptibility", axis=1)
y_test = test_data["Car Acceptibility"]


# Veri kümesinin entropisini hesaplayan fonksiyon
def entropy_hesapla(y):
    unique_classes, class_counts = np.unique(y, return_counts=True) # kaç farklı class olduğunu ve bu classlardan kaç tane olduğunu hesaplar
    probabilities = class_counts / len(y)   # her classın veri kümesindeki oranını hesaplar
    entropy_value = -np.sum(probabilities * np.log2(probabilities))  # entropi hesaplar
    return entropy_value



# Belirli bir feature için information gain hesaplayan fonksiyon
def information_gain_hesapla(X, y, feature):
    entropy_before = entropy_hesapla(y) # mevcut durumun entropisini hesaplar
    unique_values = np.unique(X[feature]) # feature'daki unique değerleri alır
    information_after = 0

    for value in unique_values:  # her bir unique değer için
        subset_y = y[X[feature] == value]  # feature'ın bu değere sahip olduğu alt kümesi
        information_after += len(subset_y) / len(y) * entropy_hesapla(subset_y)  # information'ı hesaplar

    information_gain_value = entropy_before - information_after   # information gain'i hesaplar
    return information_gain_value



# En iyi bölünme noktasını bulan fonksiyon (Gain'e bakarak)
def find_best_split(X, y):
    best_feature = None # en iyi feature için başlangıç değeri
    best_information_gain = 0 # en iyi information gain için başlangıç değeri

    for feature in X.columns: # verilen her bir feature için information gain hesaplar
        current_information_gain = information_gain_hesapla(X, y, feature)
        if current_information_gain > best_information_gain:   # current information gain, şu ana kadar en iyisiyse best feature'ı güncelle
            best_information_gain = current_information_gain
            best_feature = feature

    return best_feature # best feature'ı döndürür



# ID3 Algoritmasını kullanarak Decision Tree (Karar Ağacı) oluşturan fonksiyon.
def id3_build_tree(X, y):

    if len(np.unique(y)) == 1: # eğer veri kümesinde yalnızca bir class varsa, bir leaf node oluştur ve class'ı döndür
        return {'class': y.iloc[0]}

    best_feature = find_best_split(X, y)  # best feature'u bulur (gain'e göre)

    if best_feature is None:  # eğer best feature'u yoksa, bir leaf node oluştur ve en çok tekrar eden class'ı döndür
        return {'class': np.argmax(np.bincount(y))}

    tree = {'feature': best_feature, 'branches': {}}  # ağacı oluştur

    for value in np.unique(X[best_feature]):   # best feature'un her bir unique değeri için sub tree oluşturur
        subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        subset_y = y[X[best_feature] == value]

        tree['branches'][value] = id3_build_tree(subset_X, subset_y)  # sub tree oluşturur ve ana tree'nin alt dallarına ekler (recursive devam eder işlemler)

    return tree


# Karar ağacını kullanarak bir test data için tahmin yapar.
def predict(tree, sample):
    if 'class' in tree:  # eğer bu düğüm bir leaf node ise, class'ı döndür
        return tree['class']

    feature_value = sample[tree['feature']]
    if feature_value not in tree['branches']:  # feature'ın değeri ağaçta yoksa, "None" döndür
        return None

    branch = tree['branches'][feature_value]
    return predict(branch, sample) # tahminler ağacın geri kalanı için recursive şekilde devam eder


# Doğruluk hesaplayan fonksiyon
def accuracy_hesapla(predictions, actual):
    correct_predictions = np.sum(predictions == actual)  # doğru tahmin sayısını bulur
    total_samples = len(actual)  # toplam örnek sayısını bulur
    accuracy_percentage = (correct_predictions / total_samples) * 100  # doğruluk yüzdesini hesaplar
    return accuracy_percentage


# Karar ağacını çizen fonksiyon
def print_tree(tree, indent=0):
    if "class" in tree:
        print("|__ Sınıf:", tree["class"])
    else:
        print("|__ Özellik:", tree["feature"])
        for value, subtree in tree["branches"].items():
            print("|    " * indent + "|__ {}: ".format(value), end="")
            print_tree(subtree, indent + 1)


# Karar ağacını oluşturur
tree = id3_build_tree(X_train, y_train)

# Test veri seti için tahminler yapar
predictions = []
for _, sample in X_test.iterrows():
    predictions.append(predict(tree, sample))

# Tahminleri bir DataFrame'e dönüştürür
result_df = pd.DataFrame({'Prediction': predictions})

# Tahmin sonuçlarını "tahminler.xlsx" dosyasına yazar
result_df.to_excel("tahminler.xlsx", index=False)

# Ağacı printler
print_tree(tree)

# Test data ve Tahminler arasındaki doğruluk değerini hesaplar.
accuracy = accuracy_hesapla(predictions, y_test)
print(f"Doğruluk: {accuracy}")
