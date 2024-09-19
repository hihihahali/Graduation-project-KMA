from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Activation, Maximum, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="white")


#Dữ liệu kiểm tra là danh sách 3435 ví dụ phần mềm độc hại, mỗi ví dụ có 3514 đặc điểm API

seed_dict = pickle.load(open('feat_dict.pickle', 'rb'), encoding='latin1')
features = []
sha1 = []
for key in seed_dict:
    seed_dict[key] = seed_dict[key].toarray()[0]
    features.append(seed_dict[key])
    sha1.append(key)
feed_feat = np.stack(features)

#Tập dữ liệu huấn luyện chứa 9668 mẫu, mỗi mẫu có 3514 đặc điểm API. 5139 mẫu đầu tiên là ví dụ phần mềm độc hại, và 6294 mẫu cuối cùng là ví dụ vô hại.

class MalGAN():
    def __init__(self, model_name):
        self.apifeature_dims = 3514
        self.z_dims = 100   # nhiễu thêm vào cuối ví dụ
        self.model_name = model_name

        self.hide_layers = 256
        self.generator_layers = [self.apifeature_dims+self.z_dims, self.hide_layers, self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, self.hide_layers, 1]

        # Tạo bộ phát hiện thay thế (substitute_detector)
        self.substitute_detector = self.build_substitute_detector()
        self.optimizer = Adam(learning_rate=0.001)
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        # Tạo bộ sinh (generator)
        self.generator = self.build_generator()

        # Bộ sinh nhận đầu vào là phần mềm độc hại và nhiễu, và sinh ra các ví dụ phần mềm độc hại tấn công
        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # Đối với mô hình kết hợp, chúng ta chỉ huấn luyện bộ sinh
        self.substitute_detector.trainable = False

        # Bộ phân biệt nhận các hình ảnh sinh ra làm đầu vào và xác định tính hợp lệ
        validity = self.substitute_detector(malware_examples)

        # Mô hình kết hợp (bộ sinh và bộ phân biệt)
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        # Tải mô hình blackbox từ tệp pickle
        with open('blackbox_model.pkl', 'rb') as f:
            self.blackbox = pickle.load(f)
        
    def build_generator(self):
        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator
    
    def build_substitute_detector(self):
        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
        substitute_detector = Model(input, x, name='substitute_detector')
        substitute_detector.summary()
        return substitute_detector    

    def train(self, epochs, batch_size):
        # Tải tập dữ liệu kiểm tra (tất cả phần mềm độc hại)
        seed_dict = pickle.load(open('feat_dict.pickle', 'rb'), encoding='latin1')
        features = []
        sha1 = []
        dist_dict = {}  # [key]: hash [value]: L0 distance
        for key in seed_dict:
            seed_dict[key] = seed_dict[key].toarray()[0]
            features.append(seed_dict[key])
            sha1.append(key)
        feed_feat = np.stack(features)
        xtest_mal, ytest_mal = feed_feat, np.ones(len(feed_feat))

        # Tải tập dữ liệu huấn luyện
        train_x, train_y = load_svmlight_file("train_malgan.libsvm",
                                       n_features=3514,
                                       multilabel=False, 
                                       zero_based=False,
                                       query_id=False)
        
        
        train_x = train_x.toarray()
        xtrain_ben = train_x[5139:]
        ytrain_ben = train_y[5139:]
        xtrain_mal = train_x[0:5139]              
        ytrain_mal = train_y[0:5139]
        
        # Vì tập dữ liệu huấn luyện không cân bằng, chúng ta chọn ngẫu nhiên mẫu từ tập dữ liệu vô hại
        # và thêm vào cuối để lấp đầy khoảng trống
        idx = np.random.randint(0, xtrain_ben.shape[0], 5139 - 4529)
        add_on = xtrain_ben[idx]
        add_on_label = ytrain_ben[idx]
        xtrain_ben = np.concatenate((xtrain_ben, add_on), axis=0)
        ytrain_ben = np.concatenate((ytrain_ben, add_on_label), axis = 0)

        Test_TPR = []
        d_loss_list, g_loss_list = [], []
        
        for epoch in range(epochs):
            
            # Mỗi epoch đi qua tất cả dữ liệu trong tập huấn luyện
            start = 0
                
            for step in range(xtrain_mal.shape[0] // batch_size):
                
                # ---------------------
                #  Huấn luyện substitute_detector
                # ---------------------

                xmal_batch = xtrain_mal[start : start + batch_size]  
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))

                xben_batch = xtrain_ben[start : start + batch_size]
                start = start + batch_size
                
                # Dự đoán sử dụng mô hình blackbox
                yben_batch = self.blackbox.predict(xben_batch)

                # Sinh ra một lô ví dụ phần mềm độc hại mới
                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = self.blackbox.predict(np.ones(gen_examples.shape)*(gen_examples > 0.5))
                
                # Huấn luyện bộ phát hiện thay thế (substitute_detector)
                d_loss_real = self.substitute_detector.train_on_batch(xben_batch, yben_batch)
                d_loss_fake = self.substitute_detector.train_on_batch(gen_examples, ymal_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Huấn luyện Generator
                # ---------------------
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))
                g_loss = self.combined.train_on_batch([xmal_batch, noise], np.zeros((batch_size, 1)))

            # Sau mỗi epoch, đánh giá hiệu suất lẩn tránh trên tập dữ liệu kiểm tra
            # Thử các giá trị nhiễu khác nhau 3 lần
            for j in range(3):
                noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
                gen_examples = self.generator.predict([xtest_mal, noise])
                    
                TPR = self.blackbox.score(np.ones(gen_examples.shape)*(gen_examples > 0.5), np.ones(gen_examples.shape[0],))
            
                Test_TPR.append(TPR)
            
                transformed_to_bin = np.ones(gen_examples.shape)*(gen_examples > 0.5)
            
                pred_y_label = self.blackbox.predict(np.ones(gen_examples.shape)*(gen_examples > 0.5))

                # Xóa các ví dụ phần mềm độc hại đã lẩn tránh thành công khỏi xtest_mal
                i = 0
                print(f'Còn lại: {len(xtest_mal)}')
                while i  < pred_y_label.shape[0]:
                    if pred_y_label[i] == 0:  # Nên là 1 nhưng dự đoán là 0
                        # Tính toán khoảng cách L0 và đưa vào từ điển
                        L0 = np.sum(transformed_to_bin[i]) - np.sum(xtest_mal[i])  # chỉ chèn vào
                        dist_dict[sha1[i]] = L0  # [key]: hash [value]: L0 distance
                        xtest_mal = np.delete(xtest_mal, i, 0)
                        pred_y_label = np.delete(pred_y_label, i, 0)
                        sha1 = sha1[:i] + sha1[i+1:]
                    else:
                        i += 1
            
                print(f'Epoch {epoch+1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}, TPR: {np.mean(Test_TPR)}')
            
        # Chuyển đổi dist_dict thành DataFrame với chỉ mục tương ứng
        df = pd.DataFrame(list(dist_dict.items()), columns=['SHA1', 'L0 Distance'])

        # Lưu DataFrame thành CSV
        df.to_csv('dist_dict.csv', index=False)

# Khởi tạo và huấn luyện MalGAN
malgan = MalGAN(model_name='malgan')
malgan.train(epochs=5, batch_size=32)