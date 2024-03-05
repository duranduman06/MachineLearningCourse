import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# Veri setini okuma işlemi
col_names = ["Cinsiyet", "Yaş", "AiledeTansiyon", "FizikselAktivite", "Sigaraİçme", "AlkolTüketimi", "Tansiyon"]
train_data = pd.read_excel(r"traindata.xlsx", names=col_names)

X_train = train_data.drop("Tansiyon", axis=1)
y_train = train_data["Tansiyon"]



# Veri kümesinin entropisini hesaplayan fonksiyon
def entropy(y):
    unique_classes, class_counts = np.unique(y, return_counts=True) # kaç farklı class olduğunu ve bu classlardan kaç tane olduğunu hesaplar
    probabilities = class_counts / len(y)   # her classın veri kümesindeki oranını hesaplar
    entropy_value = -np.sum(probabilities * np.log2(probabilities))  # entropi hesaplar
    return entropy_value



# Belirli bir feature için information gain hesaplayan fonksiyon
def information_gain(X, y, feature):
    entropy_before = entropy(y) # mevcut durumun entropisini hesaplar
    unique_values = np.unique(X[feature]) # feature'daki unique değerleri alır
    information_after = 0

    for value in unique_values:  # her bir unique değer için
        subset_y = y[X[feature] == value]  # feature'ın bu değere sahip olduğu alt kümesi
        information_after += len(subset_y) / len(y) * entropy(subset_y)  # information'ı hesaplar

    information_gain_value = entropy_before - information_after   # information gain'i hesaplar
    return information_gain_value



# En iyi bölünme noktasını bulan fonksiyon (Gain'e bakarak)
def find_best_split(X, y):
    best_feature = None # en iyi feature için başlangıç değeri
    best_information_gain = 0 # en iyi information gain için başlangıç değeri

    for feature in X.columns: # verilen her bir feature için information gain hesaplar
        current_information_gain = information_gain(X, y, feature)
        if current_information_gain > best_information_gain:   # current information gain, şu ana kadar en iyisiyse best feature'ı güncelle
            best_information_gain = current_information_gain
            best_feature = feature

    return best_feature # best feature'ı döndürür



# ID3 Algoritmasını kullanarak Decision Tree (Karar Ağacı) oluşturan fonksiyon.
def build_tree(X, y):

    if len(np.unique(y)) == 1: # eğer veri kümesinde yalnızca bir class varsa, bir leaf node oluştur ve class'ı döndür
        return {'class': y.iloc[0]}

    best_feature = find_best_split(X, y)  # best feature'u bulur (gain'e göre)

    if best_feature is None:  # eğer best feature'u yoksa, bir leaf node oluştur ve en çok tekrar eden class'ı döndür
        return {'class': np.argmax(np.bincount(y))}

    tree = {'feature': best_feature, 'branches': {}}  # ağacı oluştur

    for value in np.unique(X[best_feature]):   # best feature'un her bir unique değeri için sub tree oluşturur
        subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        subset_y = y[X[best_feature] == value]

        tree['branches'][value] = build_tree(subset_X, subset_y)  # sub tree oluşturur ve ana tree'nin alt dallarına ekler (recursive devam eder işlemler)

    return tree


# Karar ağacını kullanarak bir test data için tahmin yapar.
def predict(tree, sample):
    if 'class' in tree:  # eğer bu düğüm bir leaf node ise, class'ı döndür
        return tree['class']

    feature_value = sample[tree['feature']]
    if feature_value not in tree['branches']:  # feature'ın değeri ağaçta yoksa, "0" döndür
        return 0

    branch = tree['branches'][feature_value]
    return predict(branch, sample) # tahminler ağacın geri kalanı için recursive şekilde devam eder


# Karar ağacını çizen fonksiyon
def print_tree_structure(tree, indent=0):
    if "class" in tree:
        print("|__ Sınıf:", tree["class"])
    else:
        print("|__ Özellik:", tree["feature"])
        for value, subtree in tree["branches"].items():
            print("|    " * indent + "|__ {}: ".format(value), end="")
            print_tree_structure(subtree, indent + 1)

"""
[[True Negative (TN)  False Positive (FP)]
 [False Negative (FN) True Positive (TP)]]
"""

# Confiusion Matrix'i hesaplayan fonksiyon.
def calculate_confusion_matrix_manual(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))  # true ve predicted y değerleri birleşiminden unique class elde eder.
    num_classes = len(unique_classes)  # Toplam class sayısını al.
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int) # Confusion matrixi sıfırlarla başlat.

    for true_label, predicted_label in zip(y_true, y_pred): # Her bir true ve predicted değer confusion matrixi güncelle.
        # True ve predicted sınıfların indekslerini belirleme
        true_idx = np.where(unique_classes == true_label)[0][0]
        predicted_idx = np.where(unique_classes == predicted_label)[0][0]
        confusion_matrix[true_idx, predicted_idx] += 1  # Confusion matrix'i günceller.

    return confusion_matrix # Hesaplanan confusion matrix'i döndür.

# F1 skoru hesaplama fonksiyonu
def calculate_f1_score_manual(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    f1_score_manual = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score_manual

# K-fold cross validation ve confusion matrix hesaplama.
def k_fold_cross_validation(X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) # veriyi 5 parçaya böler, 5 fold sağlar.
    fold = 1 #fold sayacı

    # Her fold için confusion matrix ve F1 skorlarını saklamak için listeler.
    confusion_matrices = []
    f1_scores_manual = []
    accuracies=[]

    # K-Fold döngüsü
    for train_index, test_index in skf.split(X, y):
        print(f"\nFold {fold}:")

        # Train ve test setlerini oluşturur.
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Karar ağacını oluşturur.
        tree = build_tree(X_train_fold, y_train_fold)

        # Test seti için tahminleri yapar.
        predictions = X_test_fold.apply(lambda x: predict(tree, x), axis=1)

        # X_test_fold'ları, gerçek değerlerini ve tahminleri Excel dosyalarına kaydetme
        result_df = pd.concat([X_test_fold, pd.DataFrame({'True_Label': y_test_fold, 'Predictions': predictions})],axis=1)
        result_df.to_excel(f'fold_{fold}_testData.xlsx', index=False)

        # Confusion matrix hesaplama
        confusion = calculate_confusion_matrix_manual(y_test_fold, predictions)
        confusion_matrices.append(confusion)

        # Accuracy hesaplama
        accuracy = np.trace(confusion) / np.sum(confusion)
        accuracies.append(accuracy)

        # F1 skoru hesaplama
        f1_manual = calculate_f1_score_manual(confusion)
        f1_scores_manual.append(f1_manual)

        #print(f"X_train length:{len(X_train_fold)}, X_test length:{len(X_test_fold)}")
        #print(f"X_Test_Fold: {X_test_fold.shape}\n{X_test_fold}\n")
        print("Confusion Matrix:")
        print(confusion)

        # TN, TP, FP, FN değerlerini bulup print eder
        tn, fp, fn, tp = confusion.ravel()
        print(f"True Negative (TN): {tn}, True Positive (TP): {tp}, False Positive (FP): {fp}, False Negative (FN): {fn}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1_manual:.4f}")

        fold += 1

    return confusion_matrices, f1_scores_manual


# Overall F1 skoru hesaplama
def calculate_average_f1_score(f1_scores):
    # F1 skorlarının toplamını alıp ve fold sayısına bölerek ortalamayı hesaplar.
    overall_f1_score = sum(f1_scores) / len(f1_scores)
    print("Overall F1 Score:", overall_f1_score)


# K-fold sonuçlarından genel doğruluk ve ortalama confusion matrix hesaplayan fonksiyon
def calculate_average_metrics(confusion_matrices):
    # doğruluk ve confusion matrixlerin ortalaması için başlangıç değerleri
    overall_accuracy = 0
    average_confusion_matrix = np.zeros_like(confusion_matrices[0], dtype=float)

    # Her bir fold için döngü
    for i, confusion_matrix in enumerate(confusion_matrices):

        # Confusion Matrix üstünden her fold için doğruluk hesaplar ve print eder.
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) # (True Positive + True Negative) / sum(Confusion Matrix) işlemi.
        overall_accuracy += accuracy

        # Average confusion matrix'e her fold'un confusion matrix'ini ekler
        average_confusion_matrix += confusion_matrix

    #ortalama accuracy değeri hesaplanır
    overall_accuracy /= len(confusion_matrices)

    # Average confusion matrix'i fold sayısına böler. Böylece average confusion matrix elde edilir.
    average_confusion_matrix /= len(confusion_matrices)

    # Elde edilen değerleri ekrana yazdırma işlemi.
    print("\nOverall:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("Average Confusion Matrix:")
    print(average_confusion_matrix)

    # TN, TP, FP, FN değerlerini bulup print eder
    tn, fp, fn, tp = average_confusion_matrix.ravel()
    print(f"True Negative (TN): {tn}, True Positive (TP): {tp}, False Positive (FP): {fp}, False Negative (FN): {fn}")


# K-fold cross validation ve confusion matrixleri hesaplama
confusion_matrices, f1_scores_manual_f1  = k_fold_cross_validation(X_train, y_train)

# Toplam accuracy ve ortalama confusion matrixi hesaplama
calculate_average_metrics(confusion_matrices)

# Toplam F1 skoru hesaplama
calculate_average_f1_score(f1_scores_manual_f1)


