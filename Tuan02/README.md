# Tuần 2 — Dự đoán giá nhà Ames

## 1. Mục tiêu

Thực nghiệm xây dựng mô hình hồi quy dự đoán giá bán nhà (`SalePrice`) trên bộ dữ liệu **House Prices: Advanced Regression Techniques** (Ames, Iowa). Mỗi căn nhà được mô tả bởi 79 biến giải thích về vị trí, chất lượng, diện tích, năm xây dựng, gara, tầng hầm, v.v.

Notebook thực nghiệm: [`lab02/code/lab02.ipynb`](lab02/code/lab02.ipynb).

## 2. Dữ liệu

| Tập dữ liệu | Số dòng | Số cột | Vai trò |
| --- | ---: | ---: | --- |
| `train.csv` | 1.460 | 81 | Huấn luyện; gồm `Id`, 79 thuộc tính và nhãn `SalePrice` |
| `test.csv` | 1.459 | 80 | Dự đoán; gồm `Id` và 79 thuộc tính, không có `SalePrice` |

Các tệp dữ liệu đặt tại `lab02/code/data/`. Mô tả đầy đủ của các thuộc tính nằm trong `data_description.txt`.

## 3. Thiết lập môi trường

Dự án sử dụng Python 3.10.11 trở lên, với các thư viện chính: pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn và Jupyter.

Từ thư mục gốc của repository:

```powershell
uv sync
cd week_2/lab02/code
uv run jupyter notebook lab02.ipynb
```

Nếu không dùng `uv`, có thể tạo môi trường ảo rồi cài đặt các gói trong `requirements.txt`. Khi chạy notebook, thư mục làm việc phải là `week_2/lab02/code` để đường dẫn tương đối `data/` được nhận diện đúng.

## 4. Quy trình thực nghiệm

### 4.1. Khảo sát và làm sạch dữ liệu

1. Đọc `train.csv` và `test.csv`; kiểm tra kiểu dữ liệu, kích thước và giá trị thiếu bằng `info()`, `isnull().sum()` và heatmap.
2. Điền giá trị thiếu của các biến số bằng giá trị trung bình, ví dụ `LotFrontage`, `GarageYrBlt`, `MasVnrArea`. Với các biến phân loại, điền mode, ví dụ `FireplaceQu`, `GarageType`, `BsmtQual` và `KitchenQual`.
3. Loại bỏ `Id` và bốn biến có tỷ lệ thiếu rất cao: `Alley`, `PoolQC`, `Fence`, `MiscFeature`.

Sau bước làm sạch, tập train có kích thước **1.460 × 76** và tập test có kích thước **1.459 × 75**.

### 4.2. Mã hóa đặc trưng

Để train và test có cùng không gian đặc trưng, hai tập được nối tạm thời trước khi mã hóa. Các biến phân loại được one-hot encoding với `drop_first=True`, sau đó loại bỏ cột trùng tên và tách lại theo mốc 1.460 dòng:

- Train sau mã hóa: **1.460 × 176** (gồm nhãn `SalePrice`);
- Test sau mã hóa: **1.459 × 175**;
- Ma trận đầu vào cho mô hình: **175 đặc trưng**.

### 4.3. Mô hình

| Mô hình | Thiết lập trong notebook |
| --- | --- |
| XGBoost Regressor | Tìm tham số bằng `RandomizedSearchCV`: 50 cấu hình, 5-fold CV, thước đo MAE âm. Cấu hình được huấn luyện lại: `n_estimators=900`, `max_depth=2`, `learning_rate=0.1`, `min_child_weight=1`, `booster='gbtree'`, `base_score=0.25`, `random_state=0`. |
| Decision Tree Regressor | `DecisionTreeRegressor(random_state=42)` với tham số mặc định còn lại. |
| Neural Network | Keras Sequential: Dense(50, ReLU) → Dense(25, ReLU) → Dense(50, ReLU) → Dense(1); khởi tạo `he_uniform`, optimizer Adamax, loss RMSE, `batch_size=10`, 1.000 epoch, `validation_split=0.25`. |

## 5. Kết quả thực nghiệm

| Hạng mục | Kết quả ghi nhận |
| --- | --- |
| Tìm kiếm XGBoost | Đã chạy 50 cấu hình × 5 folds, tương đương **250 lượt fit**. Notebook chọn cấu hình XGBoost nêu trên. |
| Dự đoán XGBoost | Mảng dự đoán có kích thước **(1459,)**; khi chạy ô cuối sẽ tạo `outputs/sample_sub_xgb.csv`. |
| Dự đoán Decision Tree | Mảng dự đoán có kích thước **(1459,)**; khi chạy ô cuối sẽ tạo `outputs/sample_sub_dt.csv`. |
| Neural Network | RMSE huấn luyện ở epoch 1.000: **19.005,66**; validation RMSE: **31.255,93**. Giá trị validation RMSE thấp nhất quan sát trong log là **28.843,87** tại epoch 983. |
| Dự đoán Neural Network | Mảng dự đoán có kích thước **(1459, 1)**; khi chạy ô cuối sẽ tạo `outputs/sample_sub_nn.csv`. |

Tập `test.csv` không chứa nhãn thật, do đó notebook hiện **không thể tính MAE/RMSE cuối cùng trên test**. Điểm số cross-validation tốt nhất của XGBoost cũng không được xuất ra hoặc lưu trong notebook; vì vậy không nên diễn giải bảng trên như một kết quả so sánh định lượng đầy đủ giữa ba mô hình. Để đánh giá công bằng, cần lưu `random_cv.best_score_` cho XGBoost và đánh giá cả ba mô hình trên cùng một validation split hoặc cùng một bộ cross-validation.

## 6. Đầu ra và tái lập

Sau khi chạy tuần tự các ô trong notebook, các đầu ra được tạo trong thư mục `lab02/code/outputs/`:

- `xgb_model.pkl`: mô hình XGBoost đã huấn luyện;
- `nn_model.keras`: mô hình mạng nơ-ron đã huấn luyện;
- `sample_sub_xgb.csv`, `sample_sub_dt.csv`, `sample_sub_nn.csv`: tệp dự đoán theo định dạng `sample_submission.csv`.

Mỗi tệp submission có hai cột `Id` và `SalePrice`, với 1.459 dòng tương ứng tập test. Vì huấn luyện mạng nơ-ron chưa cố định seed và không dùng early stopping, kết quả của lần chạy lại có thể thay đổi nhẹ; nên lưu seed, lịch sử huấn luyện và metric validation để bảo đảm khả năng tái lập tốt hơn.
