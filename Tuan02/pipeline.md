# Pipeline thực nghiệm — Lab 02

Tài liệu này mô tả chi tiết luồng xử lý của notebook [`lab02/code/lab02.ipynb`](lab02/code/lab02.ipynb), từ dữ liệu đầu vào đến các tệp dự đoán. Bài toán là hồi quy dự đoán giá bán nhà `SalePrice` trong bộ dữ liệu Ames Housing / Kaggle House Prices.

> Quy ước: đường dẫn bên dưới được hiểu tương đối từ thư mục `week_2/lab02/code/`, là thư mục cần được chọn làm working directory khi chạy notebook.

## 1. Tổng quan luồng xử lý

```text
train.csv, test.csv, sample_submission.csv
                │
                ▼
      khám phá dữ liệu và kiểm tra null
                │
                ▼
  điền giá trị thiếu + bỏ Id và 4 cột quá thiếu
                │
                ▼
     nối train/test → one-hot encoding → tách lại
                │
                ▼
       175 đặc trưng đầu vào cho ba mô hình
                │
      ┌─────────┼──────────┐
      ▼         ▼          ▼
  XGBoost   Decision Tree  Neural Network
      │         │          │
      └─────────┴──────────┘
                ▼
      3 tệp submission có 1.459 dự đoán
```

## 2. Đầu vào, thư viện và biến khởi tạo

### 2.1. Tệp đầu vào

| Tệp | Nội dung | Kích thước được notebook in ra |
| --- | --- | ---: |
| `data/train.csv` | Dữ liệu huấn luyện, có cột nhãn `SalePrice` | 1.460 × 81 |
| `data/test.csv` | Dữ liệu cần dự đoán, không có `SalePrice` | 1.459 × 80 |
| `data/sample_submission.csv` | Mẫu cột `Id`, `SalePrice` để xuất kết quả | 1.459 dòng |
| `data/data_description.txt` | Diễn giải các thuộc tính | — |

Trong `train.csv`, 81 cột gồm `Id`, 79 biến mô tả nhà và `SalePrice`. Trong `test.csv`, 80 cột gồm `Id` và 79 biến mô tả nhà.

### 2.2. Thư viện

Notebook nạp các thư viện sau:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
```

Các cell về mô hình nạp thêm `xgboost`, `RandomizedSearchCV`, `DecisionTreeRegressor` và Keras/TensorFlow. Danh sách phiên bản phụ thuộc được khai báo tại [`../requirements.txt`](../requirements.txt) và `pyproject.toml` ở thư mục gốc repository.

### 2.3. Đọc dữ liệu

```python
DATA_DIR = Path("data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
n_train = len(train)  # 1460
```

`n_train` được lưu ngay sau khi đọc dữ liệu. Giá trị này được dùng làm mốc để tách lại train/test sau khi nối hai DataFrame ở bước mã hóa.

## 3. Khảo sát dữ liệu (EDA cơ bản)

Notebook thực hiện các kiểm tra sau, chủ yếu để quan sát chứ chưa tạo đặc trưng mới:

1. In `shape` của train và test.
2. Gọi `info()` và hiển thị năm dòng đầu bằng `head()`.
3. Đếm null của từng cột bằng `isnull().sum()`.
4. Vẽ heatmap `sns.heatmap(train.isnull())` và `sns.heatmap(test.isnull())`.

Kết quả quan sát cho thấy nhiều cột phân loại có giá trị thiếu. Bốn cột `Alley`, `PoolQC`, `Fence` và `MiscFeature` có tỷ lệ thiếu lớn (trên 70% theo ghi chú trong notebook), nên được loại bỏ ở bước tiếp theo.

## 4. Làm sạch dữ liệu

### 4.1. Điền giá trị thiếu của train

Hai danh sách được khai báo riêng cho tập train:

```python
cat_col_train = [
    "FireplaceQu", "GarageType", "GarageFinish", "MasVnrType",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageQual", "GarageCond"
]
ncat_col_train = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]
```

- Với mỗi cột trong `cat_col_train`, notebook thay null bằng mode: `fillna(train[i].mode()[0])`.
- Với mỗi cột trong `ncat_col_train`, notebook thay null bằng trung bình cột: `fillna(train[j].mean())`.

`FireplaceQu` xuất hiện lặp lại trong danh sách; việc điền lần thứ hai không làm thay đổi kết quả vì các null đã được điền ở lần đầu.

### 4.2. Điền giá trị thiếu của test

Tập test có nhiều cột thiếu hơn nên danh sách được bổ sung:

- Biến phân loại bổ sung: `MSZoning`, `Utilities`, `Exterior1st`, `Exterior2nd`, `KitchenQual`, `Functional`, `SaleType`.
- Biến số bổ sung: `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`, `GarageCars`, `GarageArea`.

Quy tắc điền vẫn giống train: mode cho biến phân loại và mean cho biến số. Giá trị mode/mean của test được tính trên chính tập test, không dùng thống kê từ train.

### 4.3. Loại bỏ cột

```python
to_drop = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
for k in to_drop:
    train.drop([k], axis=1, inplace=True)
    test.drop([k], axis=1, inplace=True)
```

Sau bước này:

| DataFrame | Kích thước | Ghi chú |
| --- | ---: | --- |
| `train` | 1.460 × 76 | Vẫn gồm nhãn `SalePrice` |
| `test` | 1.459 × 75 | Không có nhãn |

Notebook vẽ lại heatmap null để kiểm tra trực quan sau khi làm sạch.

## 5. Mã hóa biến phân loại

### 5.1. Nối tạm thời train và test

```python
final_df = pd.concat([train, test], axis=0)
```

`final_df` có kích thước **2.919 × 76**. Cột `SalePrice` tồn tại trong DataFrame này và có giá trị thiếu ở phần dữ liệu test do test không có nhãn.

Mục đích nối hai tập là bảo đảm một-hot encoding tạo cùng tập cột cho train và test, kể cả khi một hạng mục chỉ xuất hiện ở một trong hai tập. Việc này chỉ dùng các giá trị đặc trưng, không dùng `SalePrice` để mã hóa.

### 5.2. One-hot encoding

Danh sách `all_cat_col` chứa 39 cột phân loại, ví dụ `MSZoning`, `Neighborhood`, `HouseStyle`, `ExterQual`, `GarageType`, `SaleType` và `SaleCondition`.

Hàm trong notebook chạy lần lượt cho từng cột:

```python
df1 = pd.get_dummies(final_df[fields], drop_first=True, dtype=int)
final_df.drop([fields], axis=1, inplace=True)
df_final = pd.concat([df_final, df1], axis=1)
```

Đặc điểm của cách cài đặt hiện tại:

- `drop_first=True`: với mỗi biến phân loại, bỏ một hạng mục tham chiếu để giảm đa cộng tuyến.
- Cột dummy nhận kiểu số nguyên `int`.
- Sau khi mã hóa, notebook gọi `final_df.loc[:, ~final_df.columns.duplicated()]` để bỏ cột có **tên** bị lặp.

Kích thước được log lại trong notebook:

| Giai đoạn | Kích thước |
| --- | ---: |
| Ngay sau one-hot encoding | 2.919 × 236 |
| Sau khi bỏ tên cột trùng | 2.919 × 176 |

> Lưu ý kỹ thuật: `pd.get_dummies` trong hàm không thêm tiền tố tên biến vào cột dummy. Vì vậy các hạng mục cùng tên ở hai biến khác nhau có thể tạo tên cột giống nhau; lệnh bỏ cột trùng sẽ giữ cột đầu tiên và bỏ cột sau. Đây là lý do pipeline thực tế có 176 cột, và cũng là điểm nên cải tiến bằng `prefix=fields` hoặc `pd.get_dummies(..., prefix=all_cat_col)`.

### 5.3. Tách lại dữ liệu cho mô hình

```python
df_train = final_df.iloc[:n_train, :].copy()
df_test = final_df.iloc[n_train:, :].copy()
df_test.drop(["SalePrice"], axis=1, inplace=True)

x_train = df_train.drop(["SalePrice"], axis=1)
y_train = df_train["SalePrice"]
```

| Biến | Kích thước | Vai trò |
| --- | ---: | --- |
| `df_train` | 1.460 × 176 | Dữ liệu đã mã hóa, gồm nhãn |
| `df_test` | 1.459 × 175 | Dữ liệu đã mã hóa để suy luận |
| `x_train` | 1.460 × 175 | Đặc trưng huấn luyện |
| `y_train` | 1.460 | Nhãn giá bán |

## 6. Nhánh mô hình 1 — XGBoost

### 6.1. Mô hình khởi tạo và tìm tham số

Notebook khởi tạo trước một `XGBRegressor` với `device="cuda"` và `n_jobs=-1`, rồi dùng làm estimator cho Randomized Search:

```python
param = {
    "n_estimators": [100, 500, 900, 1100, 1500],
    "max_depth": [2, 3, 5, 10, 15],
    "learning_rate": [0.05, 0.1, 0.15, 0.2],
    "min_child_weight": [1, 2, 3, 4],
    "booster": ["gbtree", "gblinear"],
    "base_score": [0.25, 0.5, 0.75, 1],
}

random_cv = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param,
    cv=5,
    n_iter=50,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=5,
    return_train_score=True,
    random_state=42,
)
random_cv.fit(x_train, y_train)
```

Điều này tạo **50 cấu hình × 5 folds = 250 lượt fit**. Do scikit-learn quy ước điểm cao hơn là tốt hơn, MAE được đổi dấu thành `neg_mean_absolute_error` trong quá trình tìm kiếm. Cell sau hiển thị `random_cv.best_estimator_`, nhưng notebook không lưu hay in `best_score_`.

### 6.2. Huấn luyện cấu hình được chọn

Thay vì dùng trực tiếp `random_cv.best_estimator_`, notebook tạo lại một model cố định với cấu hình sau rồi fit trên toàn bộ `x_train`:

```python
xgb_model = xgboost.XGBRegressor(
    base_score=0.25,
    booster="gbtree",
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=1,
    n_estimators=900,
    objective="reg:squarederror",
    random_state=0,
    tree_method="exact",
    n_jobs=-1,
)
xgb_model.fit(x_train, y_train)
```

Cấu hình trong mã còn đặt các hệ số điều chuẩn mặc định `reg_alpha=0`, `reg_lambda=1`, `subsample=1`, `colsample_bytree=1` cùng một số tham số mặc định khác. Model này được tuần tự hóa bằng `pickle` thành `outputs/xgb_model.pkl`.

### 6.3. Suy luận và xuất submission

```python
pred_xgb = xgb_model.predict(df_test)      # shape: (1459,)
sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_df["SalePrice"] = pred_xgb
output_file = OUTPUT_DIR / "sample_sub_xgb.csv"
sub_df.to_csv(output_file, index=False)
```

## 7. Nhánh mô hình 2 — Decision Tree

```python
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)

pred_dt = dt_model.predict(df_test)        # shape: (1459,)
sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_df["SalePrice"] = pred_dt
output_file = OUTPUT_DIR / "sample_sub_dt.csv"
sub_df.to_csv(output_file, index=False)
```

Đây là baseline cây quyết định hồi quy, không có bước tuning hay validation riêng. Do không giới hạn `max_depth`, cây có thể khớp rất sát dữ liệu huấn luyện và có nguy cơ overfit.

## 8. Nhánh mô hình 3 — Neural Network

### 8.1. Hàm loss RMSE

Notebook định nghĩa RMSE bằng Keras Ops (có fallback sang Keras Backend):

```python
def root_mean_squared_error(y_true, y_pred):
    return k.sqrt(k.mean(k.square(y_pred - y_true)))
```

### 8.2. Kiến trúc và huấn luyện

```python
nn_model = Sequential()
nn_model.add(Dense(50, kernel_initializer="he_uniform", activation="relu",
                   input_dim=x_train.shape[1]))
nn_model.add(Dense(25, kernel_initializer="he_uniform", activation="relu"))
nn_model.add(Dense(50, kernel_initializer="he_uniform", activation="relu"))
nn_model.add(Dense(1, kernel_initializer="he_uniform"))
nn_model.compile(loss=root_mean_squared_error, optimizer="Adamax")

nn_model.fit(
    x_train.values, y_train.values,
    validation_split=0.25,
    batch_size=10,
    epochs=1000,
)
```

Thông số thực nghiệm:

| Thành phần | Giá trị |
| --- | --- |
| Số đầu vào | 175 |
| Hidden layers | 50 → 25 → 50 neuron, ReLU |
| Đầu ra | 1 neuron tuyến tính dự đoán `SalePrice` |
| Khởi tạo trọng số | `he_uniform` |
| Optimizer | Adamax |
| Loss/metric hiển thị | RMSE tự định nghĩa |
| Validation split | 25% của `x_train` |
| Batch size / epochs | 10 / 1.000 |

Log đã lưu trong notebook ghi nhận ở epoch 1.000: training RMSE **19.005,66** và validation RMSE **31.255,93**. Giá trị validation RMSE thấp nhất xuất hiện trong log là **28.843,87** tại epoch 983. Model cuối epoch (không phải checkpoint tốt nhất) được lưu tại `outputs/nn_model.keras`.

### 8.3. Suy luận và xuất submission

```python
pred_nn = nn_model.predict(df_test)        # shape: (1459, 1)
sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_df["SalePrice"] = pred_nn.reshape(-1)
output_file = OUTPUT_DIR / "sample_sub_nn.csv"
sub_df.to_csv(output_file, index=False)
```

`reshape(-1)` chuyển mảng hai chiều thành vector trước khi gán vào cột `SalePrice`.

## 9. Đầu ra của pipeline

Khi cell đọc dữ liệu được chạy, `OUTPUT_DIR.mkdir(exist_ok=True)` sẽ tạo thư mục `lab02/code/outputs/` nếu thư mục này chưa tồn tại. Khi các cell lưu tương ứng được chạy, thư mục này sẽ có các tệp sau:

| Tệp | Nguồn | Nội dung |
| --- | --- | --- |
| `outputs/xgb_model.pkl` | XGBoost | Mô hình đã được pickle |
| `outputs/nn_model.keras` | Neural Network | Mô hình Keras ở epoch cuối |
| `outputs/sample_sub_xgb.csv` | XGBoost | `Id`, dự đoán `SalePrice` cho 1.459 mẫu |
| `outputs/sample_sub_dt.csv` | Decision Tree | `Id`, dự đoán `SalePrice` cho 1.459 mẫu |
| `outputs/sample_sub_nn.csv` | Neural Network | `Id`, dự đoán `SalePrice` cho 1.459 mẫu |

Các submission không được sinh sẵn trong repository; chúng chỉ xuất hiện sau khi chạy các cell xuất tệp tương ứng. Mỗi cell lưu sẽ in đường dẫn tuyệt đối của tệp vừa tạo để người chạy có thể mở đúng vị trí.

## 10. Giới hạn và đề xuất cải thiện

1. **Chưa có phép đánh giá chung cho ba mô hình.** Test không có nhãn, còn notebook chỉ lưu log validation của Neural Network. Nên dùng cùng một hold-out split hoặc `KFold` và báo cáo MAE, RMSE/RMSLE cho mọi mô hình.
2. **Cần lưu kết quả tuning.** Thêm `print(-random_cv.best_score_)`, `random_cv.best_params_` và lưu bảng `cv_results_` để đối chiếu các lần chạy.
3. **Cần chuẩn hóa cho Neural Network.** Các biến diện tích, năm xây dựng và giá nhà có thang đo rất khác nhau nhưng hiện được đưa trực tiếp vào mạng. Có thể dùng `StandardScaler` trong `Pipeline`, fit trên phần train của mỗi fold.
4. **Cần early stopping và checkpoint.** Hiện mạng luôn chạy 1.000 epoch và chỉ lưu epoch cuối, dù validation RMSE tốt nhất xuất hiện sớm hơn. Có thể dùng `EarlyStopping(restore_best_weights=True)` và `ModelCheckpoint`.
5. **Cần cố định tính tái lập.** Thiết lập seed cho Python, NumPy, TensorFlow/Keras và XGBoost; đồng thời lưu phiên bản thư viện và phần cứng thực thi.
6. **Cần cải thiện one-hot encoding.** Dùng `OneHotEncoder(handle_unknown="ignore")` trong `ColumnTransformer` hoặc đặt prefix khi dùng `get_dummies`, tránh bỏ đặc trưng chỉ vì trùng tên hạng mục.
7. **Cần cân nhắc biến đổi nhãn.** `SalePrice` thường lệch phải; thử huấn luyện trên `np.log1p(SalePrice)` và hoàn nguyên bằng `np.expm1` khi xuất dự đoán, rồi đánh giá bằng metric phù hợp với cuộc thi như RMSLE.

## 11. Cách chạy lại

Từ thư mục gốc repository:

```powershell
uv sync
cd week_2/lab02/code
uv run jupyter notebook lab02.ipynb
```

Trong Jupyter, chọn **Run All Cells** hoặc chạy tuần tự từ đầu đến cuối. Không chạy lại riêng một cell one-hot encoding khi `final_df` đã bị thay đổi bởi các cell trước, vì hàm mã hóa thao tác `inplace=True` trên DataFrame này.
