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

* 🗣️ **Cách chém gió:** "Dạ nhóm không chọn bừa. Hệ thống đã tự test ngẫu nhiên 4 mô hình: Baseline, Logistic Reg, Random Forest và Gradient Boosting thông qua **Cross-validation 5-Fold**. Gradient Boosting đạt chuẩn vô địch với `AUC = 0.96`. Nó cũng phù hợp nhất cho dạng dữ liệu bảng biểu (Tabular Data) như thế này ạ."

---

## 🎯 CÂU HỎI 3: DỊCH CÁC CON SỐ ĐÁNH GIÁ (METRICS)

Nhìn vào bảng điểm tốt nghiệp của Model:
> `Accuracy = 89% | Precision = 85% | Recall = 80% | ROC-AUC = 0.96`

👨‍🏫 **Hội đồng hỏi:** *"Ứng dụng cái mớ này vào app GameNect ra sao?"*

1. **✅ Accuracy (Độ chính xác 89%):** Xác suất cực cao. Đưa 10 cặp game thủ bất kỳ, AI phán trúng 9 cặp sẽ Match hay Dislike.
2. **🏹 Precision (Sự chắc cú 85%):** Rất dễ hiểu thưa thầy, khi AI đã "Gợi ý" ai đó lên màn hình điện thoại, thì **85%** là người dùng sẽ ấn nút LIKE!
3. **🔍 Recall (Độ phủ 80%):** Trong đại dương người chơi kia có 100 cặp đôi cực kỳ hợp nhau, thì AI của em sẽ quét và tóm được 80 cặp lên bờ, tránh bỏ sót nhân tài.
4. **⚖️ ROC-AUC (0.96):** Đây là chứng chỉ minh chứng AI có khả năng rạch ròi 100% giữa việc *Thích/Ghét*, không có chuyện nó bói mò kiểu 50-50.

---

## ⚔️ CÂU HỎI 4: BẢO VỆ "QUY TẮC CHỦ QUAN"

👨‍🏫 **Câu hỏi sát thủ:** *"Các em dùng luật (Tuổi gần nhau, Rank bằng nhau) để sinh Data Synthetic. Làm sao chứng minh luật đó là đúng chuẩn Khoa Học?"*

Hãy đập lại liền 3 ý sau:
1. **Bài toán Cold-Start (Khởi động lạnh):** App mới ra không thể có thuật toán Lọc Cộng Tác vì lúc đó trắng thông tin. Buộc phải dùng Knowledge-based (Trí thức nền) để qua giai đoạn sơ khai.
2. **Quy luật Tâm lý - Homophily (Sự tương cận):** Logic tuổi tác gần nhau dựa trên thuyết Tâm Lý Học, con người có xu hướng dễ kết bầy với những người có chung đặc điểm vòng đời.
3. **Thuyết Thiết kế Game - Bartle Taxonomy:** Việc ép Rank và Phong cách (Competitive/Casual) dựa trên thuyết phân loại người chơi. Khép một "Try Hard Rank Phèn" với một "Gà mờ đi check-in" sẽ dẫn đến ức chế chửi thề làm hỏng môi trường game thưa thầy.

> Rút gọn lại, mấy luật thô cứng đó chỉ là "Sách giáo khoa" cho Model học hỏi ban đầu. Càng về sau nó càng học được luật mềm dẻo thông qua Data Real!

---

## 🧬 CÂU HỎI 5: KỸ THUẬT MA TRẬN VECTOR HÓA

👨‍🏫 **Hội đồng hỏi:** *"Vector hóa Feature là nạp vô AI như nào?"*

* Nhóm áp dụng **Pairwise Vectorization**. Biến mọi chữ trong Profile thành một cụm Vector **62 chiều** rạch ròi.
* **Biến độ lệch (Delta):** Em trừ số tuổi, trừ win-rate đi, vì AI dễ hiểu sự "Chênh Lệch" hơn là hiểu một số thô nguyên bản.
* **Tọa độ Map (Haversine):** Vĩ/Kinh độ được cho qua công thức đại số Haversine để tạo lại thành khoảng cách thực tế (ví dụ: 12 km).
* **Chuẩn hóa (Z-Score Standard Scaler):** Nếu quăng số "Km" đứng cùng số "Tuổi", Model sẽ thiên vị biến số lớn. Nhóm xài Standard Scaler để kéo mọi thông số trong vũ trụ 62 chiều này quy về một đơn vị đo lường chung nhất (Mean=0, Std=1).

> **🎉 CHỐT LẠI: "Dạ trình tự AI của em minh bạch từ lúc lấy số tới lúc trả ra điểm Match, xin hết thưa hội đồng!"**
