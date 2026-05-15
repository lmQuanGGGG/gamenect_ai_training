<div align="center">
  
# 🎓 KỊCH BẢN BẢO VỆ ĐỒ ÁN GAMENECT
### 🤖 Phần: Training AI Machine Learning
*Tài liệu tổng hợp các cách lập luận, giải thích con số và phản biện hội đồng chấm thi.*

</div>

<br>

---

## 📊 CÂU HỎI 1: BẢNG SỐ LIỆU TRAINING

Khi rớt Terminal ra bảng này:
```text
Total pairs    :   6313
Real (weighted):   2614 (41.4%)
Synthetic      :   3699 (58.6%) weight=0.5
```

👨‍🏫 **Hội đồng hỏi:** *"Real và Synthetic là gì? Sao lại trộn vào nhau?"*
* 🟢 **Real (Dữ liệu thật):** 2.614 cặp đôi thu từ lịch sử Quẹt/Match CÓ THẬT của người dùng Firebase. Nó có giá trị cao nhất vì là "người thật việc thật".
* 🟡 **Synthetic (Dữ liệu giả lập):** 3.699 cặp do nhóm tự đặt luật tạo ra. Lý do? Vì App mới lập, Data thật chưa đủ lớn để AI khôn lên được. Nhóm gán trọng số `weight=0.5` cho Data giả để Model luôn nghiêng về trải nghiệm thực tế. Đây gọi là **Phương pháp Hybrid (Lai)**.

---

## 🏆 CÂU HỎI 2: LỰA CHỌN THUẬT TOÁN

👨‍🏫 **Hội đồng hỏi:** *"Tại sao chọn Gradient Boosting?"*

* 🛡️ **Hướng trả lời:** "Dạ hệ thống đã tự động thử nghiệm song song 4 mô hình: Baseline, Logistic Regression, Random Forest và Gradient Boosting thông qua **Cross-validation 5-Fold**. Kết quả Gradient Boosting thể hiện sự vượt trội với `AUC = 0.96`. Hơn nữa, thuật toán dạng Cây (Tree-based) này cũng có hiệu năng trích xuất tính năng chuẩn xác nhất đối với dạng dữ liệu bảng biểu nhiều chiều (Tabular Data) mà đồ án đang ứng dụng ạ."

---

## 🎯 CÂU HỎI 3: DỊCH CÁC CON SỐ ĐÁNH GIÁ (METRICS)

Nhìn vào bảng điểm tốt nghiệp của Model:
> `Accuracy = 89% | Precision = 85% | Recall = 80% | ROC-AUC = 0.96`

👨‍🏫 **Hội đồng hỏi:** *"Các chỉ số này thể hiện điều gì đối với trải nghiệm thực tế trên app GameNect?"*

1. **✅ Accuracy (Độ chính xác 89%):** Xác suất đánh giá tổn thể. Với 10 cặp game thủ bất kỳ, AI có khả năng phán đoán chính xác 9 cặp sẽ đi đến Quyết định Match hay Bỏ qua.
2. **🏹 Precision (Độ chuẩn xác 85%):** Đảm bảo chất lượng trải nghiệm: Trong số những người mà AI đề xuất lên bảng tin của người dùng, có tới **85%** khả năng người dùng sẽ đánh giá tích cực và thực hiện thao tác Quẹt Phải (Like).
3. **🔍 Recall (Độ phủ 80%):** Tránh bỏ sót các gợi ý tiềm năng. Giả sử có 100 cặp đôi cực kỳ hợp nhau đang online trên hệ thống, AI sẽ quét và bao phủ được 80 cặp, khai thác tối đa mạng lưới người dùng hợp cạ.
4. **⚖️ ROC-AUC (0.96):** Đây là thông số thống kê minh chứng AI có năng lực phân tách chuẩn xác giữa tập "Tương thích" và "Không tương thích". Mô hình đạt trạng thái ổn định lý tưởng và không hề phân loại ngẫu nhiên.

---

## ⚔️ CÂU HỎI 4: BẢO VỆ "QUY TẮC CHỦ QUAN"

👨‍🏫 **Câu hỏi tình huống:** *"Các em sử dụng luật tự đặt (Tuổi gần nhau, Rank bằng nhau) để sinh Data Synthetic. Làm sao để chứng minh các luật rẽ nhánh đó có cơ sở khoa học và không làm hỏng tính chất của AI?"*

* 🛡️ **Phương pháp luận bảo vệ:**
1. **Khắc phục điểm yếu Cold-Start (Khởi động lạnh):** Ứng dụng ở giai đoạn khởi tạo không thể áp dụng thuật toán Lọc Cộng Tác (Collaborative Filtering) vì rào cản chưa có thông tin tương tác. Nhóm giải quyết bằng cách dùng Knowledge-based (Trí thức nền) làm bệ phóng qua vùng sơ khai.
2. **Nguyên lý Tâm lý học xã hội - Homophily (Sự tương cận):** Thiết lập độ tuổi gần nhau bắt nguồn từ học thuyết Tâm Lý Học, cho rằng con người luôn có xu hướng dễ dàng bao dung và kết bầy với những người có chung giai đoạn vòng đời.
3. **Hệ thống phân loại người chơi - Bartle Taxonomy:** Việc quy hoạch tệp Rank và Phong cách (Competitive/Casual) dựa trên nguyên tắc Game Design. Khớp nối một người chơi hạng "Chuyên nghiệp" với một người chơi hạng "Trải nghiệm" sẽ dễ dẫn đến trạng thái ức chế tương tác, đi ngược lại lợi ích của ứng dụng GameNect.

> Tóm lại, các bộ luật ràng buộc này chỉ đóng vai trò là "Kiến thức Mẫu" (Prior Knowledge) cho Model làm quen ban đầu. Càng về sau, khối lượng Data Thực tế đổ về càng lớn, thuật toán sẽ tự động uyển chuyển theo xu hướng của đám đông thưa hội đồng.

---

## 🧬 CÂU HỎI 5: KỸ THUẬT MA TRẬN VECTOR HÓA

👨‍🏫 **Hội đồng hỏi:** *"Vector hóa Feature là nguyên lý gì? Nhóm thực thi như thế nào?"*

* Nhóm áp dụng **Pairwise Vectorization**. Biến mọi chữ trong Profile thành một cụm Vector **62 chiều** rạch ròi.
* **Biến độ lệch (Delta):** Em trừ số tuổi, trừ win-rate đi, vì AI dễ hiểu sự "Chênh Lệch" hơn là hiểu một số thô nguyên bản.
* **Tọa độ Map (Haversine):** Vĩ/Kinh độ được cho qua công thức đại số Haversine để tạo lại thành khoảng cách thực tế (ví dụ: 12 km).
* **Chuẩn hóa (Z-Score Standard Scaler):** Nếu quăng số "Km" đứng cùng số "Tuổi", Model sẽ thiên vị biến số lớn. Nhóm xài Standard Scaler để kéo mọi thông số trong vũ trụ 62 chiều này quy về một đơn vị đo lường chung nhất (Mean=0, Std=1).

> **🎉 TẠM CHỐT PHẦN 1: "Dạ trình tự Pipeline của nhóm em hoàn chỉnh từ bước chuẩn hóa đến phân loại, em xin kết thúc phần trình bày kỹ thuật cơ sở ạ!"**

---

<br>

<div align="center">
  
# 💻 PHẦN 2: CÁC CÂU HỎI TIẾP CẬN TỪ GÓC ĐỘ KỸ SƯ PHẦN MỀM VÀ HỆ THỐNG
*(Các luận điểm kết hợp giữa System Architecture và Machine Learning)*

</div>

<br>

## ⚙️ CÂU HỎI 6: TẠI SAO LỰA CHỌN MACHINE LEARNING THAY VÌ LUẬT IF-ELSE THÔNG THƯỜNG?

👨‍🏫 **Câu hỏi phản biện:** *"Tại sao hệ thống không sử dụng các câu lệnh rẽ nhánh (if-else constraints) như `Tuổi gần nhau && Cùng rank thì Match` để tiết kiệm tài nguyên mà lại sinh ra Machine Learning?"*

* 🛡️ **Hướng trả lời:** 
"Dạ thưa hội đồng, phương pháp lập trình tuần tự (Rule-based) chỉ giải quyết được các bài toán tĩnh, có tính phân cực Tuyệt Đối (Black-or-White). Ví dụ nếu dùng Rule-based, khi người dùng chênh lệch 1 mức Rank, hệ thống lập tức loại bỏ cơ hội Match của họ. 

Tuy nhiên, tâm lý con người mang tính **Đánh đổi (Trade-offs)**. Một cặp đôi có thể chênh lệch mức Rank, nhưng bù lại khoảng cách địa lý của họ vô cùng gần và họ share chung một lượng game khổng lồ. 
Mô hình Machine Learning (Gradient Boosting) có khả năng tính toán ra quy luật bù trừ (Latent patterns) này dựa trên 62 thông số để đưa ra một số liệu Xác suất Tương thích (Probability Score). Điều này giúp hệ thống linh hoạt hơn, bao dung hơn và đặc biệt là hệ thống tự cải thiện độ chính xác theo thời gian khi có Dataset lớn hơn (Data-driven pattern), việc mà mã nguồn mở if-else tốn diện tích không thể tự làm được."

---

## 🚀 CÂU HỎI 7: HIỆU NĂNG VÀ KIẾN TRÚC TRIỂN KHAI THỰC TẾ

👨‍🏫 **Câu hỏi phản biện:** *"Mô hình AI yêu cầu tính toán sâu, nếu tích hợp thẳng vào App Flutter trên thiết bị di động có gây ra hiện tượng tràn RAM, nóng máy hoặc làm chậm quá trình phản hồi không?"*

* 🛡️ **Hướng trả lời:** 
"Dạ với định hướng Kỹ Sư Phần Mềm, nhóm em thiết kế kiến trúc hệ thống theo mô hình **Microservices** và tách biệt hoàn toàn Client khỏi AI processing. 
Cụ thể, Client (Flutter App) giữ vai trò UI thuần túy. Cụm AI được cấu hình dưới dạng một RESTful API server độc lập bằng thư viện **FastAPI** (Python). 
Để đạt độ trễ tiệm cận 0 (Zero-latency), mô hình Classifier (file `.pkl`) được hệ thống **preload trực tiếp lên RAM (In-memory loaded)** ngay chu kỳ khởi động. Khi Client bắn request sang, hệ thống đưa ra quyết định mà không cần đọc/ghi ổ đĩa cứng, nên tốc độ phản hồi trả về App chưa tốn đến 15 milliseconds. Trải nghiệm vuốt (Swipe) của người dùng được đảm bảo hoàn toàn mượt mà."

---

## 🔄 CÂU HỎI 8: BÀI TOÁN 'MODEL DRIFT' VÀ VÒNG ĐỜI MLOps

👨‍🏫 **Câu hỏi phản biện:** *"Mô hình hiện tại được đánh giá cao, nhưng sau 3 tháng nữa khi xu hướng game thay đổi hoặc data bị 'Data Drift', mô hình của em có bị phán đoán sai lệch đi không?"*

* 🛡️ **Hướng trả lời:** 
"Dạ đây là một vấn đề cực kỳ phổ biến trong vòng đời của AI. Để khắc phục, nhóm em đã tích hợp mô hình **MLOps (Machine Learning Operations)** kết hợp với CI/CD Pipeline. 

Thay vì train một lần rồi bỏ đó, hệ thống được nhóm cấu hình một luồng **Automated Retraining** trên máy chủ GitHub Actions. Mỗi ngày, tiến trình sẽ tự động mở cổng giao tiếp lên Firebase để cập nhật các tương tác vuốt và match trực tiếp của người dùng vào đêm hôm trước. Hệ thống chạy lại quá trình Optimize, sau đó đẩy mô hình hoàn thiện mới nhất qua cổng API của Server (Dạng Hot-reload) mà không làm gián đoạn hệ thống. Nhờ bộ tuần hoàn tự động này, mô hình AI của nhóm em là một dạng AI sống, luôn tự thay đổi trọng số phân loại kịp thời theo xu hướng (trend) của chính hệ thống."

---

## ⚖️ CÂU HỎI 9: XỬ LÝ LỆCH DỮ LIỆU (IMBALANCED DATA)

👨‍🏫 **Câu hỏi phản biện:** *"Thực tế số người bị Quẹt Trái (Từ chối) luôn cao gấp rất nhiều lần số người được Quẹt Phải (Match), em xử lý lượng dữ liệu bị thiếu cân bằng này như thế nào?"*

* 🛡️ **Hướng trả lời:** 
"Dạ thưa hội đồng, việc Negative Labels (No Match) áp đảo là bản chất của các Recommender Systems. Nhóm em không dùng `Accuracy` (Độ phân loại chuẩn) làm thước đo chính, vì nó bị che mắt bởi nghịch lý Accuracy Paradox. Thay vào đó, nhóm chú trọng vào **ROC-AUC (Hiện tại đạt 0.96)** và **F1-Score**.
Trực tiếp ở khâu thiết kế Model, thuật toán của em sử dụng tham số `class_weight` kết hợp với kỹ thuật cây quyết định của thuật toán Gradient Boosting để phạt nặng những trường hợp dự đoán sai các mẫu hiếm (Cặp Match). Nó giúp Model bắt rớt cực kỳ nhạy và hiểu thói quen của người quẹt."

---

## 🔍 CÂU HỎI 10: TỐC ĐỘ TÍNH TOÁN QUY MÔ LỚN (SCALABILITY)

👨‍🏫 **Câu hỏi phản biện:** *"Giả sử nền tảng có 1,000,000 người dùng tích cực. Nếu 1 người yêu cầu gợi ý, hệ thống mang 62 thông số của người đó đi đọ chéo với cả 1 triệu người kia thì server có sập không?"*

* 🛡️ **Hướng trả lời:**
"Dạ hoàn toàn không sập ạ. Để giải quyết rủi ro Bottle-neck này, thuật toán ở Backend API thiết kế một mạng lưới **Candidate Filtering (Lọc ứng viên thô)** trước khi truyền cho Machine Learning.
Khi có yêu cầu từ người dùng, hệ thống chỉ khoanh vùng (Query Bounds) vị trí địa lý trong bán kính Max Distance và giới hạn độ tuổi mà người dùng thiết lập. Truy vấn thô (Pre-filter) này loại bỏ đi 99% lượng user nằm ngoài bộ lọc. Sau đó, Model AI mới áp dụng tính toán ma trận độ tương thích cho danh sách khoảng 500 Candidates cuối cùng sót lại. Việc này tiết kiệm tài nguyên Cloud cực lớn ạ."

---

## 📈 CÂU HỎI 11: GIẢI THÍCH Ý NGHĨA 5 BIỂU ĐỒ REPORT (REPORT CHARTS)

👨‍🏫 **Hội đồng hỏi:** *"Trong phần báo cáo có 5 biểu đồ đánh giá model, em hãy giải thích ngắn gọn ý nghĩa của từng biểu đồ và nó chứng minh điều gì cho hệ thống của em?"*

* 🛡️ **Hướng trả lời:**
1. **Biểu đồ `learning_curve.png` (Đường cong học tập):** Dùng để chứng minh mô hình **không bị bệnh 'học vẹt' (Overfitting)**. Đường điểm số của tập Train và tập Validation hội tụ sát lại với nhau khi lượng dữ liệu tăng lên. Điều này chứng tỏ AI thực sự 'hiểu' được quy luật ghép đôi thay vì chỉ học thuộc lòng đáp án.
2. **Biểu đồ `model_evaluation.png` (Ma trận nhầm lẫn & ROC):** Đây là bài thi thực tế của AI. Ở Confusion Matrix, hai ô True Positive và True Negative có số lượng lớn nhất, cho thấy AI đoán trúng phóc các cặp Match và Không Match. Chỉ số ROC-AUC đạt mức rất cao (0.91 đến 0.96) khẳng định AI phân tách người dùng cực kỳ dứt khoát.
3. **Biểu đồ `score_distribution.png` (Phân bổ điểm số):** Thể hiện **sự dứt khoát** của AI. Hai đỉnh đồ thị tách biệt rõ ràng: Đỉnh màu xanh (Không hợp) dồn về mốc điểm rất thấp, đỉnh màu cam (Hợp nhau) tập trung ở mốc điểm rất cao. AI không hề lưỡng lự hay chấm điểm mập mờ.
4. **Biểu đồ `shap_importance.png` (Xếp hạng độ quan trọng):** Giúp 'giải mã' hộp đen AI. Biểu đồ xếp hạng các yếu tố quan trọng nhất quyết định việc ghép đôi (thanh dài nhất nằm trên cùng như Khoảng cách, Phong cách chơi, Chênh lệch tuổi). Điều này chứng tỏ AI tư duy cực kỳ logic giống hệt tâm lý con người.
5. **Biểu đồ `shap_beeswarm.png` (Tác động chi tiết):** Cho thấy **chiều tác động** của từng yếu tố. Chấm đỏ là giá trị cao, xanh là thấp. Ví dụ: 'Khoảng cách' màu đỏ nằm dạt về bên trái trục 0 (Khoảng cách càng xa thì điểm Match càng bị trừ nặng). Hệ thống hoạt động hoàn hảo và thực tế.

---

## 🧩 CÂU HỎI 12: CHỈ SỐ COMPAT_FACTOR_SCORE LÀ GÌ? TẠI SAO LÀ 11 YẾU TỐ?

👨‍🏫 **Hội đồng hỏi:** *"Biến `compat_factor_score` đóng vai trò quan trọng nhất (18.2%). Biến này là gì và tại sao em lại thiết kế nó bao gồm 11 yếu tố thay vì 7 yếu tố như lúc đầu?"*

* 🛡️ **Hướng trả lời:**
"Dạ thưa thầy cô, biến `compat_factor_score` là một **chỉ số tương thích nền tảng (từ 0 đến 1)** đóng vai trò như một 'chiếc mỏ neo' định hướng mạnh mẽ cho AI trước khi nó xét đến các thông số phức tạp khác. Bản chất của nó là lấy trung bình cộng của 11 điều kiện cốt lõi nhất.

Ở phiên bản Baseline (dữ liệu giả), em chỉ dùng **7 yếu tố cơ bản** (Khớp giới tính, Khớp độ tuổi, Khớp khoảng cách, Hợp phong cách chơi, Trình độ tương đương, Khoảng cách dưới 50km, Chênh lệch dưới 7 tuổi).

Tuy nhiên, khi chuyển sang train với **Dữ liệu thật (Real Data)** từ Firebase ở phiên bản AI v3.0, hành vi người dùng đa dạng và phức tạp hơn rất nhiều. Do đó, em đã tiến hành Feature Engineering để nâng cấp bộ 'Mỏ neo' này lên thành **11 yếu tố**, bổ sung thêm 4 ràng buộc cực kỳ thực tế:
8. **Chơi chung tựa game:** Có ít nhất 1 game chung (điều kiện tiên quyết để chơi chung).
9. **Cùng mục đích:** Cùng tìm người leo rank, hoặc tìm bạn tâm giao.
10. **Tài khoản xác thực (Verified):** Tăng độ tin cậy và an toàn (trust/safety).
11. **Trạng thái Hoạt động:** Cả 2 đều có hoạt động trong 30 ngày đổ lại (tránh ghép với tài khoản đã 'chết').

Việc nâng cấp này giúp AI chặn ngay các tài khoản ảo hoặc không còn hoạt động, tăng tính thực tiễn khi triển khai ra hệ thống thật ạ."

---

## ❄️ CÂU HỎI 13: BÀI TOÁN COLD-START (KHỞI ĐỘNG LẠNH)

👨‍🏫 **Hội đồng hỏi:** *"Hệ thống của em là gợi ý ghép đôi (Recommender System). Vậy với một User VỪA MỚI TẠO TÀI KHOẢN, chưa hề có lịch sử Quẹt (Like/Dislike) hay Match, thì AI lấy cơ sở ở đâu để gợi ý cho họ?"*

* 🛡️ **Hướng trả lời:**
"Dạ đây chính là bài toán **Cold-Start Problem** cực kỳ kinh điển. Để giải quyết, mô hình AI của nhóm em sử dụng phương pháp **Content-Based Filtering (Lọc theo nội dung)** kết hợp với **Hybrid Dataset**.
Khi user mới vào, AI sẽ không dùng lịch sử tương tác (vì chưa có), mà nó sẽ bóc tách ngay **54 thông số tĩnh (Static Features)** từ profile họ vừa khai báo (như Độ tuổi, Khoảng cách GPS, Giới tính tìm kiếm, Phong cách chơi game, Tựa game yêu thích). Từ đó, AI sử dụng bộ dữ liệu Synthetic (những quy tắc chuẩn mực mà nhóm đã mồi sẵn với trọng số 0.5) để tìm ra những người có Profile tương đồng nhất. Nhờ vậy, ngay từ lần quẹt đầu tiên, user đã thấy những ứng viên rất sát với nhu cầu của mình ạ."

---

## 🔒 CÂU HỎI 14: BẢO MẬT VÀ QUYỀN RIÊNG TƯ DỮ LIỆU (DATA PRIVACY)

👨‍🏫 **Hội đồng hỏi:** *"Để AI tính toán được, App Flutter phải gửi toàn bộ dữ liệu nhạy cảm (Tọa độ GPS, Tuổi, Giới tính) của người dùng qua API của server FastAPI. Vậy em xử lý vấn đề bảo mật dữ liệu này như thế nào?"*

* 🛡️ **Hướng trả lời:**
"Dạ thưa thầy cô, nhóm em thiết kế kiến trúc AI hoàn toàn độc lập và **Không lưu trạng thái (Stateless)**. 
Khi App gọi API để xin gợi ý, dữ liệu chỉ tồn tại trên RAM của server FastAPI trong đúng vài phần trăm giây để model thực hiện tính toán ma trận tương thích. Ngay khi tính toán xong và trả về điểm số, toàn bộ Payload dữ liệu bị hủy ngay lập tức, **Server AI không lưu trữ bất kỳ log hay record nào vào Database**.
Ngoài ra, dữ liệu tọa độ (Latitude/Longitude) truyền lên chỉ dùng để thư viện `geopy` tính ra khoảng cách tuyệt đối (Distance_km) để nạp vào AI, chứ hệ thống không hề track vị trí cụ thể của người dùng ạ."

---

## ⚖️ CÂU HỎI 15: SAI SỐ AI (FALSE POSITIVE VS FALSE NEGATIVE)

👨‍🏫 **Hội đồng hỏi:** *"Trong AI luôn có sai số. Theo em, việc AI ghép nhầm 2 người không hợp nhau (False Positive) và việc AI bỏ sót 2 người rất hợp nhau (False Negative), cái nào gây hậu quả nặng nề hơn cho App GameNect?"*

* 🛡️ **Hướng trả lời:**
"Dạ theo quan điểm phát triển sản phẩm của nhóm em, việc **Bỏ sót (False Negative) gây hậu quả nặng nề hơn**. 
Nếu AI gợi ý nhầm một người không hợp (False Positive), người dùng chỉ tốn 1 giây để Quẹt Trái (Bỏ qua) - việc này rất bình thường trên các app hẹn hò. 
Nhưng nếu AI bỏ sót một 'tri kỷ' chơi game cực kỳ hợp (False Negative), chúng ta đang làm mất đi giá trị cốt lõi của mạng xã hội là sự kết nối. Vì vậy, trong quá trình Tune Model, ngoài độ chính xác, nhóm em đặc biệt tối ưu chỉ số **Recall (Độ phủ) lên mức 80%**, để đảm bảo 'Thà quét nhầm còn hơn bỏ sót' những ứng viên tiềm năng ạ."
