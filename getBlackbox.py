from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Xử lý dữ liệu từ train_blackbox.libsvm
data_x, data_y = load_svmlight_file("train_blackbox.libsvm",
                                     n_features=3514,
                                     multilabel=False, 
                                     zero_based=False,
                                     query_id=False)

data_x = data_x.toarray()

# Chia dữ liệu thành malicious và benign
x_mal = data_x[0:5206]
y_mal = data_y[0:5206]
x_ben = data_x[5206:]
y_ben = data_y[5206:]

# Chia dữ liệu mal và ben thành huấn luyện và kiểm tra
x_mal_train, x_mal_test, y_mal_train, y_mal_test = train_test_split(x_mal, y_mal, test_size=0.2, random_state=0)
x_ben_train, x_ben_test, y_ben_train, y_ben_test = train_test_split(x_ben, y_ben, test_size=0.2, random_state=0)

# Cân bằng dữ liệu huấn luyện
# Nếu số lượng benign ít hơn, lấy thêm mẫu để cân bằng với malware
if len(x_ben_train) < len(x_mal_train):
    idx = np.random.randint(0, len(x_ben_train), len(x_mal_train))
    x_ben_train = x_ben_train[idx]
    y_ben_train = y_ben_train[idx]

# Kết hợp dữ liệu huấn luyện đã cân bằng
X_train_balanced = np.concatenate((x_mal_train, x_ben_train), axis=0)
y_train_balanced = np.concatenate((y_mal_train, y_ben_train), axis=0)

# Huấn luyện mô hình BlackBox
def black_box(train_data, train_labels):
    rfc = RandomForestClassifier(n_estimators=50, random_state=1, oob_score=True, max_depth=8)
    rfc.fit(train_data, train_labels)
    return rfc

blackbox = black_box(X_train_balanced, y_train_balanced)

# Đánh giá mô hình trên dữ liệu kiểm tra
X_test = np.concatenate((x_mal_test, x_ben_test), axis=0)
y_test = np.concatenate((y_mal_test, y_ben_test), axis=0)
y_pred = blackbox.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

# Lưu mô hình BlackBox đã huấn luyện
blackbox_file = 'blackbox_model.pkl'
with open(blackbox_file, 'wb') as file:
    pickle.dump(blackbox, file)
