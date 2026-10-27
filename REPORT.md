# BÁO CÁO KỸ THUẬT: NGHIÊN CỨU VÀ ỨNG DỤNG MẠNG TÍCH CHẬP (CNN) CHO BÀI TOÁN PHÂN LOẠI VÀ HỒI QUY TRÊN BỘ DỮ LIỆU KHUÔN MẶT UTKFace

---

## 1. Giới thiệu & Đặt vấn đề

Phân tích thuộc tính khuôn mặt (Facial Attribute Analysis) từ ảnh chụp tự nhiên là một trong những bài toán nền tảng của Thị giác Máy tính (Computer Vision). Một hệ thống phân tích khuôn mặt toàn diện phải giải quyết đồng thời hai bản chất toán học khác nhau trên cùng một không gian điểm ảnh đầu vào $\mathcal{X} \in \mathbb{R}^{C \times H \times W}$:

1. **Bài toán Phân loại (Classification)**: Dự đoán biến mục tiêu rời rạc định danh $y \in \{0, 1, \dots, C-1\}$, bao gồm:
   - **Phân loại nhị phân (Binary Classification)**: Dự đoán Giới tính (`Male` / `Female`).
   - **Phân loại đa lớp (Multi-class Classification)**: Dự đoán Sắc tộc (`White`, `Black`, `Asian`, `Indian`, `Others`).
2. **Bài toán Hồi quy (Regression)**: Dự đoán biến mục tiêu liên tục $y \in \mathbb{R}^+$, cụ thể là ước lượng Độ tuổi thực tế (`1 – 116` tuổi).

Báo cáo này trình bày phương pháp tiếp cận **Thiết kế Mô hình Dẫn dắt bởi Dữ liệu (Data-Driven Architecture Design)**: xuất phát từ các số liệu thống kê thực nghiệm ở **Giai đoạn 1 (`UTKFace_1.ipynb`)** để suy luận và chứng minh tính đúng đắn của mọi quyết định lựa chọn kiến trúc mạng, hàm kích hoạt (Activation Function), hàm mất mát (Loss Function), phương pháp khởi tạo trọng số (Weight Initialization) và chiến lược tối ưu hóa ở **Giai đoạn 2 (`UTKFace_2.ipynb`)**.

---

## 2. Khảo sát & Tiền xử lý Dữ liệu (Exploratory Data Analysis)

### 2.1. Toàn vẹn dữ liệu & Loại bỏ nhiễu nhãn

Bộ dữ liệu **UTKFace** (`jangedoo/utkface-new`) gồm các ảnh khuôn mặt đã được căn chỉnh và cắt khung (`aligned & cropped face chips`). Qua khảo sát cấu trúc lưu trữ trong `UTKFace_1.ipynb`:
- Ba thư mục con trong gói dữ liệu (`UTKFace/`, `utkface_aligned_cropped/UTKFace/`, `crop_part1/`) chứa các bản sao trùng lặp tên file. Việc chỉ thu thập từ thư mục gốc `UTKFace/` (`23,708` file) là bắt buộc nhằm ngăn chặn hiện tượng một bức ảnh xuất hiện đồng thời ở cả tập Train và tập Test (**Data Leakage**).
- Quá trình kiểm tra định dạng nhãn `[age]_[gender]_[race]_[date&time].jpg` phát hiện **3 file lỗi thiếu trường `race`**:
  - `61_1_20170109142408075.jpg.chip.jpg`
  - `61_1_20170109150557335.jpg.chip.jpg`
  - `39_1_20170116174525125.jpg.chip.jpg`
- Tập dữ liệu sạch cuối cùng gồm **$N = 23,705$ ảnh**.

### 2.2. Phân tích thống kê biến mục tiêu Hồi quy (`age`)

Kết quả đo lường phân phối độ tuổi trên `23,705` mẫu:
- **Miền giá trị**: $[1, 116]$ tuổi; **Trung bình ($\mu$)**: `33.30` tuổi; **Độ lệch chuẩn ($\sigma$)**: `19.89` tuổi.
- **Tứ phân vị**: $Q_1 = 23.0$, $Q_2 (\text{Median}) = 29.0$, $Q_3 = 45.0 \implies \text{IQR} = Q_3 - Q_1 = 22.0$ tuổi.
- **Độ lệch (Skewness)**: $\gamma_1 = 0.622 > 0$, phản ánh phân phối **lệch phải (Right-skewed)** với mật độ tập trung dày ở lứa tuổi `20 – 35` và một đỉnh phụ ở nhóm trẻ sơ sinh `1 – 2` tuổi.
- **Phát hiện điểm ngoại lai (Outliers)**: Áp dụng quy tắc Tukey:
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR} = 45.0 + 1.5 \times 22.0 = 78.0 \text{ tuổi}$$
  Có **`696` bức ảnh (`2.94%` tổng dữ liệu)** có độ tuổi nằm trong khoảng `[79, 116]`.

![Phân phối biến Age](assets/01_eda_age_distribution.png)

### 2.3. Phân tích thống kê biến mục tiêu Phân loại (`gender` & `race`)

Chỉ số mất cân bằng lớp (**Imbalance Ratio — $\text{IR}$**) của một bài toán phân loại $C$ lớp được định nghĩa bằng tỷ số giữa số mẫu của lớp lớn nhất và lớp nhỏ nhất:
$$\text{IR} = \frac{\max_{c \in \{0,\dots,C-1\}} N_c}{\min_{c \in \{0,\dots,C-1\}} N_c}$$

1. **Đối với `gender` ($C = 2$)**:
   - `Male (0)`: $N_0 = 12,391$ (`52.27%`)
   - `Female (1)`: $N_1 = 11,314$ (`47.73%`)
   - $\text{IR}_{\text{gender}} = \frac{12,391}{11,314} \approx 1.10\text{x} \implies$ Phân bố cân bằng.

2. **Đối với `race` ($C = 5$)**:
   - `White (0)`: $N_0 = 10,078$ (`42.51%`)
   - `Black (1)`: $N_1 = 4,526$ (`19.09%`)
   - `Indian (3)`: $N_3 = 3,975$ (`16.77%`)
   - `Asian (2)`: $N_2 = 3,434$ (`14.49%`)
   - `Others (4)`: $N_4 = 1,692$ (`7.14%`)
   - $\text{IR}_{\text{race}} = \frac{10,078}{1,692} \approx 5.96\text{x} \implies$ Phân bố mất cân bằng nghiêm trọng (nhóm `White` gấp gần 6 lần nhóm `Others`).

Để cân bằng đóng góp đạo hàm của cả 5 lớp trong quá trình tối ưu hóa, trọng số của mỗi lớp $c$ được xác định sao cho $w_c \cdot N_c = \frac{N_{\text{total}}}{C}$:
$$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
Thay số với $N_{\text{total}} = 23,705$ và $C = 5$:
- $w_0 (\text{White}) = \frac{23,705}{5 \times 10,078} = 0.470$
- $w_1 (\text{Black}) = \frac{23,705}{5 \times 4,526} = 1.048$
- $w_2 (\text{Asian}) = \frac{23,705}{5 \times 3,434} = 1.381$
- $w_3 (\text{Indian}) = \frac{23,705}{5 \times 3,975} = 1.193$
- $w_4 (\text{Others}) = \frac{23,705}{5 \times 1,692} = 2.802$

![Phân bố Gender và Race](assets/02_eda_gender_race_distribution.png)

### 2.4. Phân tích tương quan đa biến & Chiến lược phân chia dữ liệu

Khảo sát chéo độ tuổi theo giới tính và sắc tộc cho thấy:
- Độ tuổi trung bình của Nam giới (`35.70` tuổi, trung vị `34.0`) cao hơn Nữ giới (`30.68` tuổi, trung vị `26.0`) tròn `5` năm.
- Độ tuổi trung bình của nhóm `White` (`37.98` tuổi, trung vị `35.0`) cao hơn nhóm `Others` (`23.17` tuổi, trung vị `25.0`) tới `14.81` năm.

Nếu chia ngẫu nhiên thuần túy (`Random Split`), tỷ lệ các nhóm thiểu số như `Nữ - Others` hay `Nam - Asian` có thể bị lệch giữa tập Train và Test. Do đó, dự án thiết lập biến phân tầng tổ hợp `stratify_col = gender_race` (`2 × 5 = 10` nhóm) và chia theo tỷ lệ chuẩn **`70% Train (16,593 mẫu) : 15% Validation (3,556 mẫu) : 15% Test (3,556 mẫu)`**.

![Phân tích Đa biến Age theo Gender và Race](assets/03_eda_multivariate_age_gender_race.png)

### 2.5. Đặc trưng không gian điểm ảnh & Chuẩn hóa kênh màu

Thống kê trên `1,000` ảnh ngẫu nhiên cho thấy 100% ảnh có kích thước `200 × 200 × 3 (RGB)` với:
$$\mu_{RGB} = [0.5926, 0.4526, 0.3864], \quad \sigma_{RGB} = [0.2573, 0.2288, 0.2246]$$
Khác với bộ thông số ImageNet ($\mu = [0.485, 0.456, 0.406]$ vốn tính trên ảnh phong cảnh, đồ vật, động vật tự nhiên), ảnh khuôn mặt UTKFace có giá trị trung bình kênh Đỏ ($\mu_R = 0.5926$) cao hơn hẳn kênh Xanh lá ($\mu_G = 0.4526$) và Xanh dương ($\mu_B = 0.3864$) do đặc tính phản xạ của sắc tố melanin và hemoglobin dưới da. Chuẩn hóa từng kênh $c \in \{R, G, B\}$ theo công thức:
$$x_c^{\text{norm}} = \frac{x_c - \mu_c}{\sigma_c}$$
giúp đưa phân phối đầu vào về trung bình `0` và phương sai `1`, tăng tốc độ hội tụ của các lớp `Conv2d` ngay từ Epoch đầu tiên.

---

## 3. Thiết kế Kiến trúc Mạng CNN & Cơ sở Toán học

### 3.1. Kiến trúc Backbone Tích chập Tự thiết kế (`UTKFaceBackbone`)

Mạng cơ sở gồm **4 khối tích chập (`ConvBlock`)** nối tiếp nhau, mỗi khối thực hiện chuỗi phép biến đổi:
$$\text{Conv2d}(k=3\times 3, s=1, p=1) \longrightarrow \text{BatchNorm2d} \longrightarrow \text{LeakyReLU}(0.01) \longrightarrow \text{MaxPool2d}(2\times 2)$$

- **Sự tiến hóa của kích thước Tensor và Trường tiếp nhận (Receptive Field)**:
  - **Khối 1 (`3 → 32` kênh, `128×128 → 64×64`)**: Học các bộ lọc Gabor bậc thấp (cạnh ngang/dọc, độ dốc sáng tối, sắc tố da cục bộ).
  - **Khối 2 (`32 → 64` kênh, `64×64 → 32×32`)**: Tổ hợp các cạnh thành các chi tiết hình thái bộ phận (khóe mắt, chân mày, cánh mũi, viền môi, nếp nhăn nhỏ).
  - **Khối 3 (`64 → 128` kênh, `32×32 → 16×16`)**: Trích xuất cấu trúc từng vùng khuôn mặt (hốc mắt, gò má, rãnh mũi - má, xương quai hàm).
  - **Khối 4 (`128 → 256` kênh, `16×16 → 8×8`)**: Tổng hợp biểu diễn ngữ nghĩa toàn bộ khuôn mặt.
- **Global Average Pooling (`AdaptiveAvgPool2d(1, 1)`) & Fully-Connected Layer**:
  - Thay vì trải phẳng trực tiếp tensor `256 × 8 × 8 = 16,384` chiều (vốn làm phát sinh hơn `2 triệu` tham số ở lớp `Linear` kế tiếp và gây Overfitting nặng), `AdaptiveAvgPool2d(1, 1)` cô đọng mỗi bản đồ đặc trưng `8×8` thành `1` giá trị trung bình không gian, tạo ra vector đặc trưng `256` chiều.
  - Đi qua tiếp 1 lớp ẩn `Linear(256 → 128) + BatchNorm1d(128) + LeakyReLU(0.01) + Dropout(0.3)`, tổng số tham số của toàn bộ mô hình chỉ là **`422,049` tham số**, cực kỳ gọn nhẹ nhưng giàu sức biểu diễn.

### 3.2. Lý giải lựa chọn Hàm Kích hoạt, Khởi tạo Trọng số & Hàm Mất mát

1. **Tại sao chọn `LeakyReLU(0.01)` ở các lớp ẩn thay vì `ReLU`, `Sigmoid` hay `Tanh`?**
   - `Sigmoid` và `Tanh` có đạo hàm tối đa lần lượt là `0.25` và `1.0`, dễ bị bão hòa về `0` ở hai đầu cực trị gây triệt tiêu đạo hàm (**Vanishing Gradient**) khi xếp chồng nhiều lớp.
   - `ReLU` ($f(x) = \max(0, x)$) khắc phục được bão hòa miền dương nhưng có đạo hàm bằng `0` trên toàn bộ miền âm ($x < 0$). Trong bài toán hồi quy tuổi có Outliers lớn, một bước cập nhật trọng số mạnh có thể đẩy tổ hợp tuyến tính của một số bộ lọc sang miền âm đối với mọi mẫu dữ liệu, khiến nơ-ron đó không bao giờ cập nhật lại được nữa (**Dying ReLU**).
   - `LeakyReLU` định nghĩa:
     $$f(x) = \begin{cases} x & \text{nếu } x \ge 0 \\ 0.01x & \text{nếu } x < 0 \end{cases}$$
     luôn duy trì hệ số góc nhỏ `0.01` ở miền âm, bảo toàn dòng chảy Gradient xuyên suốt 4 khối tích chập.

2. **Tại sao khởi tạo trọng số bằng Kaiming He Normal (`kaiming_normal_`)?**
   - Với lớp tích chập có kích thước bộ lọc $k \times k$ và số kênh đầu vào $C_{\text{in}}$ (tổng số kết nối vào một nơ-ron là $n_{\text{in}} = C_{\text{in}} \cdot k^2$), nếu khởi tạo trọng số quá nhỏ thì tín hiệu tắt dần về `0`; nếu quá lớn thì tín hiệu bùng nổ qua từng lớp.
   - Phương pháp Kaiming He lấy mẫu trọng số từ phân phối chuẩn có trung bình bằng `0` và phương sai:
     $$W \sim \mathcal{N}\left(0, \sigma^2\right), \quad \sigma = \sqrt{\frac{2}{(1 + a^2) \cdot n_{\text{in}}}}$$
     (với $a = 0.01$ là độ dốc âm của `LeakyReLU`), đảm bảo phương sai đầu ra của mỗi lớp tích chập luôn xấp xỉ phương sai đầu vào ($\text{Var}(y_l) \approx \text{Var}(y_{l-1})$).

3. **Sự khác biệt cốt lõi giữa Output Layer và Loss Function của 3 bài toán**:

| Bài toán | Lớp Output & Kích hoạt | Công thức Hàm Mất mát (Loss Function) |
| :--- | :--- | :--- |
| **1. Binary Classification (`Gender`)** | `Linear(128, 1)` xuất logit $z \in \mathbb{R}$<br/>Xác suất: $\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$ | **Binary Cross-Entropy (`BCEWithLogitsLoss`)**:<br/>$$\mathcal{L}_{\text{BCE}}(y, \hat{p}) = -\left[y \ln(\hat{p}) + (1 - y)\ln(1 - \hat{p})\right]$$ |
| **2. Multi-class Classification (`Race`)** | `Linear(128, 5)` xuất vector $\mathbf{z} \in \mathbb{R}^5$<br/>Xác suất: $\hat{p}_c = \frac{e^{z_c}}{\sum_{j=0}^{4} e^{z_j}}$ ($\sum_{c=0}^4 \hat{p}_c = 1$) | **Weighted Categorical Cross-Entropy**:<br/>$$\mathcal{L}_{\text{WCE}}(y, \mathbf{z}) = -w_y \ln\left(\frac{e^{z_y}}{\sum_{j=0}^{4} e^{z_j}}\right)$$ |
| **3. Continuous Regression (`Age`)** | `Linear(128, 1)` + **Identity** ($\hat{y} = z \in \mathbb{R}$) | **Huber / Smooth L1 Loss ($\beta = 5.0$)**:<br/>$$\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} \frac{1}{2\beta}(y - \hat{y})^2 & \text{nếu } |y - \hat{y}| < \beta \\ |y - \hat{y}| - \frac{1}{2}\beta & \text{ngược lại} \end{cases}$$ |

---

## 4. Phân tích Kết quả Thực nghiệm & Đánh giá Chuyên sâu

### 4.1. Đánh giá Mô hình 1: `BinaryGenderCNN`

- **Hội tụ**: Sau 15 Epochs huấn luyện, `Train Loss` giảm từ `0.5865` xuống `0.2348` (`Train Acc` tăng từ `68.73%` lên `90.18%`). Trọng số tốt nhất đạt được tại **Epoch 12** (`Val Loss = 0.2510`, `Val Acc = 88.75%`).
- **Hiệu năng trên Tập Test (`3,556` ảnh)**:
  - **Accuracy**: **`89.57%`** (`3,185 / 3,556` mẫu dự đoán chính xác).
  - **Macro F1-Score**: **`89.53%`**.
  - **Phân tích lỗi từ Ma trận Nhầm lẫn**: Trong số `371` ảnh dự đoán sai (`155` ảnh Nam nhầm thành Nữ và `216` ảnh Nữ nhầm thành Nam), phần lớn rơi vào nhóm trẻ sơ sinh (`1 – 3` tuổi) nơi các đặc trưng giới tính thứ cấp trên khuôn mặt (xương hàm, râu, đường chân mày, kiểu tóc) chưa biểu hiện rõ ràng.

![Kết quả BinaryGenderCNN](assets/05_cnn_binary_gender_results.png)

---

### 4.2. Đánh giá Mô hình 2: `MultiClassRaceCNN`

- **Hội tụ**: Nhờ `ReduceLROnPlateau` tự động giảm Learning Rate khi `Val Loss` dao động nhẹ ở các Epoch 3, 7, 11, mô hình đạt **Best Val Macro F1 = `65.41%`** tại **Epoch 10**.
- **Hiệu năng trên Tập Test (`3,556` ảnh)**:
  - **Test Accuracy**: **`72.13%`** | **Test Macro F1-Score**: **`65.01%`** | **Weighted F1-Score**: **`72.92%`**.
- **Đánh giá tác động của Trọng số Cân bằng Lớp (`Class Weights`)**:
  - Đối với 3 lớp có đặc trưng hình thái rõ nét (`White`, `Black`, `Asian`), mô hình đạt F1-Score rất tốt từ **`72.46%` đến `82.38%`** (trong đó `Asian` đạt Precision cao nhất **`88.06%`**).
  - Đối với lớp thiểu số `Others` (`7.14%` dữ liệu, $w_4 = 2.802$): Trọng số phạt cao giúp mô hình nhận diện đúng `113 / 254` mẫu Test (**Recall = `44.49%`**), bảo vệ thành công nhóm thiểu số khỏi bị mô hình bỏ qua. Tuy nhiên, việc nhân hệ số phạt `2.802` cũng khiến ranh giới quyết định của lớp `Others` mở rộng, thu hút `139` mẫu `Indian` và `97` mẫu `White` có tông màu da trung gian, dẫn đến **Precision = `23.49%`**. Đây là sự đánh đổi kinh điển giữa **Recall lớp thiểu số** và **Precision tổng thể** khi dùng Inverse Class Frequency Weighting.

![Kết quả MultiClassRaceCNN](assets/06_cnn_multiclass_race_results.png)

---

### 4.3. Đánh giá Mô hình 3: `AgeRegressionCNN`

- **Hội tụ**: Sai số `Val MAE` giảm hơn một nửa: từ `15.26` năm (Epoch 1) xuống **`7.42` năm (Epoch 15)**. Tại Epoch 15, `Train Huber Loss` (`5.4462`) và `Val Huber Loss` (`5.3980`) gần như bằng nhau tuyệt đối, chứng minh sự kết hợp giữa `BatchNorm2d`, `Dropout(0.3)`, `Data Augmentation` và `SmoothL1Loss(beta=5.0)` đã ngăn chặn hoàn toàn hiện tượng Overfitting.
- **Hiệu năng trên Tập Test (`3,556` ảnh)**:
  - **Test MAE**: **`7.53` năm**
  - **Test RMSE**: **`10.46` năm**
  - **Hệ số xác định ($R^2$)**: **`0.7261`**
- **Phân tích sai số dư (`Residual Analysis`)**:
  - Khoảng cách giữa `RMSE` (`10.46`) và `MAE` (`7.53`) là `2.93` năm — một mức chênh lệch nhỏ đối với dữ liệu trải dài `116` năm tuổi, chứng tỏ không có các dự đoán sai lệch cực đoan.
  - Trên biểu đồ Scatter Plot, ở vùng tuổi từ `1` đến `70`, đám mây điểm bám rất sát đường chéo lý tưởng $y = \hat{y}$. Ở vùng tuổi rất cao (`> 80` tuổi), điểm dự đoán có xu hướng nằm thấp hơn nhẹ so với đường chéo (dự đoán khoảng `65 – 85` tuổi cho người `90 – 100` tuổi) do mật độ mẫu ở vùng này rất thưa (`< 3%`).

![Kết quả AgeRegressionCNN](assets/07_cnn_age_regression_results.png)

---

### 4.4. Đánh giá Mô hình 4: `MultiTaskFaceResNet` (Transfer Learning + Multi-Task Learning)

Mô hình sử dụng Backbone **`ResNet-18`** đã huấn luyện trước trên `1.28 triệu` ảnh ImageNet, thay thế lớp phân loại cuối bằng 3 nhánh đầu ra độc lập (`gender_head`, `race_head`, `age_head`) và tối ưu hóa bằng hàm mất mát tổng hợp (**Joint Loss**):
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{gender}} + 1.0 \cdot \mathcal{L}_{\text{race}} + 10.0 \cdot \mathcal{L}_{\text{Huber}}\left(\frac{y_{\text{age}}}{100}, \hat{y}_{\text{age\_scaled}}\right)$$

Chỉ sau **8 Epochs** (đạt tốt nhất tại **Epoch 5**), `MultiTaskFaceResNet` đã vượt trội hơn toàn bộ 3 mô hình Custom CNN độc lập trên mọi thước đo:
- **Gender Accuracy**: `89.57%` $\longrightarrow$ **`92.15%`** (`+2.58%`)
- **Gender Macro F1**: `89.53%` $\longrightarrow$ **`92.15%`** (`+2.62%`)
- **Race Accuracy**: `72.13%` $\longrightarrow$ **`78.26%`** (`+6.13%`)
- **Race Macro F1**: `65.01%` $\longrightarrow$ **`73.52%`** (**`+8.51%`**)
- **Age Test MAE**: `7.53` năm $\longrightarrow$ **`5.49` năm** (**Giảm `2.04` năm sai số, tương đương cải thiện `27.1%`**)
- **Age Test RMSE**: `10.46` năm $\longrightarrow$ **`7.67` năm** (**Giảm `2.79` năm**)
- **Age Test $R^2$**: `0.7261` $\longrightarrow$ **`0.8531`** (**`+0.1270`**)

**Ba nguyên nhân cốt lõi tạo nên sự bứt phá của `MultiTaskFaceResNet`**:
1. **Liên kết tắt (Residual Skip Connections $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$)**: Cho phép mạng sâu 18 lớp truyền thẳng đạo hàm về các lớp đầu tiên mà không bị suy giảm, trích xuất được các vi cấu trúc trên da mặt (vết chân chim, độ chảy xệ cơ mặt, sắc thái da tinh tế).
2. **Chuyển giao tri thức (Transfer Learning)**: Các tầng tích chập sớm đã sở hữu sẵn bộ lọc cạnh, góc và kết cấu bề mặt hoàn thiện từ ImageNet, giúp mô hình đạt `Val Gender Acc = 80.65%` ngay từ Epoch 1.
3. **Bổ trợ biểu diễn trong Học Đa nhiệm (Shared Representation & Implicit Regularization)**: Trong sinh học con người, biểu hiện lão hóa da phụ thuộc chặt chẽ vào cả giới tính và sắc tộc (ví dụ: cấu trúc collagen và sắc tố melanin ảnh hưởng đến độ tuổi xuất hiện nếp nhăn). Khi cả 3 nhánh cùng cập nhật trọng số lên một Backbone duy nhất, vector đặc trưng `512` chiều buộc phải mã hóa thông tin tổng hợp của cả 3 thuộc tính, đóng vai trò như một bộ điều chuẩn tự nhiên giúp tăng khả năng tổng quát hóa trên tập Test.

---

## 5. Kết luận & Hướng phát triển

1. **Kết luận**:
   - Việc thực hiện **EDA bài bản trước khi xây dựng mô hình** đã giúp xác định chính xác: (a) bộ chuẩn hóa kênh màu riêng của UTKFace, (b) sự cần thiết của `SmoothL1Loss` trước `696` mẫu Outliers độ tuổi, và (c) cơ chế `Weighted CrossEntropyLoss` cùng thước đo `Macro F1-Score` trước tỷ lệ mất cân bằng `5.96x` của nhãn sắc tộc.
   - Ba mô hình **Custom CNN (`422K` tham số)** tự thiết kế đạt hiệu năng nền tảng vững chắc (`Gender Acc = 89.57%`, `Race Acc = 72.13%`, `Age MAE = 7.53` năm).
   - Mô hình hợp nhất **`MultiTaskFaceResNet` (`ResNet-18`)** nâng tầm hiệu năng lên mức cao (`Gender Acc = 92.15%`, `Race Macro F1 = 73.52%`, `Age MAE = 5.49` năm, $R^2 = 0.8531$) đồng thời rút gọn chi phí lưu trữ và suy luận từ 3 mạng riêng biệt về **1 mạng duy nhất**.

2. **Hướng phát triển tiếp theo**:
   - **Làm mịn trọng số lớp (Smoothed Class Weights)**: Áp dụng công thức $w_c^{\text{smooth}} = \left(\frac{N_{\text{total}}}{C \times N_c}\right)^\alpha$ với $\alpha \in [0.5, 0.7]$ hoặc kết hợp **Focal Loss** để cải thiện Precision của lớp `Others` mà vẫn giữ được Recall cao.
   - **Tích hợp bộ phát hiện khuôn mặt tự động (Face Detection Pre-processor)**: Gắn thêm tầng cắt khuôn mặt tự động (**MTCNN** hoặc **RetinaFace**) ở đầu pipeline suy luận để xử lý các bức ảnh chụp góc rộng ngoài thực tế trước khi đưa vào mạng CNN.
   - **Ordinal Regression / Label Distribution Learning (LDL) cho bài toán Tuổi**: Kết hợp `SmoothL1Loss` với phân phối nhãn mềm (Gaussian Label Distribution quanh tuổi thật) để giảm `Age MAE` xuống ngưỡng `< 4.8` năm.
