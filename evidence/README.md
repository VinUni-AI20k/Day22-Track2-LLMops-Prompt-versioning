# RAGAS Evaluation Analysis: Dương Văn Hiệp - 2A202600052

Dựa trên báo cáo đánh giá của RAGAS (`03_ragas_report.json`), chúng ta có thể rút ra một số nhận xét như sau:

1. **Faithfulness (Độ trung thực):** Prompt V2 (đạt `0.8358`) vượt trội hơn hẳn so với Prompt V1 (`0.6077`) và đã vượt qua mức mục tiêu tối thiểu là `0.8`. Điều này cho thấy Prompt V2 với cấu trúc chi tiết hơn đã ép LLM tuân thủ chặt chẽ theo các context được cung cấp (retrieved context), hạn chế tối đa việc hallucination (bịa đặt thông tin).
2. **Answer Relevancy & Context Recall:** Prompt V1 có nhỉnh hơn một chút ở các chỉ số này. Tuy nhiên, sự đánh đổi này là hoàn toàn xứng đáng vì trong các hệ thống RAG thực tế (production), độ trung thực (Faithfulness) luôn là yếu tố quan trọng nhất để đảm bảo an toàn thông tin.
3. **Kết luận:** Prompt V2 là lựa chọn an toàn và tối ưu hơn cho hệ thống RAG này vì nó định hướng câu trả lời của LLM bám sát vào tri thức nội bộ một cách đáng tin cậy.
