# Research-and-Development-of-a-Multimodal-Data-Mining-System-for-English-and-Vietnamese-Documents
📘 Nghiên cứu và phát triển hệ thống khai thác dữ liệu đa phương thức trong tài liệu tiếng Anh và tiếng Việt dựa trên mô hình trí tuệ nhân tạo đa tác tử

Hệ thống này được thiết kế nhằm phân tích – tách trích – hiểu – truy vấn – hỏi đáp trên các tài liệu học thuật đa phương thức (văn bản, hình ảnh, bảng, sơ đồ, công thức Toán), hỗ trợ tiếng Việt và tiếng Anh, dựa trên kiến trúc AI Multi-Agent (đa tác tử) và các mô hình hiện đại.

Mục tiêu:

    📄 Hiểu nội dung văn bản
    🧮 Nhận diện & chuyển đổi công thức (OCR → LaTeX)
    🖼 Phân tích hình ảnh và sinh mô tả
    🔍 Xây dựng cơ sở tri thức từ tài liệu
    💬 Trả lời câu hỏi bằng RAG (Retrieval-Augmented Generation)
    ⚡ Đa ngôn ngữ (Việt/Anh)
    🧠 Hoạt động theo mô hình tác tử LangGraph

🚀 1. Kiến trúc tổng thể

Hệ thống gồm các thành phần:

    1. Docling — xử lý tài liệu
    - Trích văn bản PDF/docx
    - Tách bảng, hình, chú thích
    - Phân tích cấu trúc (mục lục, section, heading)
    - Chuẩn hoá đầu ra thành Docling JSON
    
    2. Pix2Tex — OCR công thức sang LaTeX
    - Nhận diện công thức trong ảnh
    - Xuất LaTeX chính xác
    - Tối ưu cho tài liệu khoa học
    
    3. Qwen3-VL — sinh caption cho hình ảnh
    - Nhận diện nội dung ảnh
    - Tạo mô tả ngữ nghĩa giàu thông tin
    - Hỗ trợ tiếng Việt & tiếng Anh
    - Tích hợp vào quá trình RAG

    4. Mô hình nhúng (Embedding)
    - M3 Embedding → Embedding văn bản
    - MathBERT → Embedding công thức Toán
    - Dùng cho truy vấn semantic search
    - Tạo vector store qua FAISS

    5. Vector database — FAISS
    - Lưu trữ vector (text + công thức)
    - Hỗ trợ RAG tốc độ cao

    6. Qwen3 (LLM) — tác tử hội thoại & tổng hợp
    - Tổng hợp kết quả truy xuất
    - Trả lời tiếng Việt hoặc tiếng Anh
    - Hỗ trợ reasoning (enable_thinking)

    7. LangGraph — hệ thống đa tác tử

Hệ thống được thiết kế dưới dạng các tác tử:

Tác tử	Vai trò
🎯 Orchestrator Agent	Điều phối pipeline, phát hiện ngôn ngữ, xác định loại truy vấn
📄 Text Retrieval Agent	Truy xuất văn bản từ FAISS
🧮 Formula Retrieval Agent	Truy xuất công thức bằng MathBERT
🖼 Vision Caption Agent	Gọi Qwen3-VL sinh mô tả hình ảnh
🧪 Fusion Agent	Hợp nhất kết quả truy vấn (Text + Formula + Vision)
💬 Answer Agent	Dùng Qwen3 sinh câu trả lời (RAG)

📥 2. Quy trình hoạt động
    (1) Người dùng upload tài liệu
    → Docling phân tích → sinh text, tables, figures, formulas

    (2) Tạo tác tử (agent)
    → Hệ thống xây dựng FAISS index
    → Tạo các embedding text + công thức
    → Nhúng hình ảnh (Qwen3-VL captioning)

    (3) Người dùng đặt câu hỏi (VN/EN)
    → Orchestrator phát hiện ngôn ngữ
    → Xác định cần truy xuất: văn bản, công thức hay hình ảnh
    → Chuyển yêu cầu cho Retrieval Agents

    (4) Hợp nhất kết quả
    → Fusion Agent chuẩn hoá, xếp hạng, trộn nhiều nguồn

    (5) Qwen3 sinh câu trả lời (RAG)
    → Dựa trên dữ liệu truy xuất
    → Trả bằng tiếng Việt hoặc tiếng Anh, tuỳ thói quen ngôn ngữ của người dùng

🛠 3. Công nghệ sử dụng
    Trích xuất PDF: Docling
    Nhận dạng công thức: Pix2Tex
    Caption ảnh: Qwen3-VL
    Embedding text: M3 Embedding
    Embedding công thức: MathBERT
    Vector DB: FAISS
    LLM trả lời: Qwen3
    Multi-Agent Orchestration: LangGraph
    Giao diện: Upload → Tạo agent → Chat

4. Kết quả đạt được
Bộ dữ liệu được sử dụng để thử nghiệm là Test-A trong bộ dữ liệu SPIQA. Các câu hỏi và trả lời được gom nhóm theo bài báo. Kết quả của các kịch bản thử nghiệm và kết quả tốt nhất của các mô hình sử dụng trong bài báo được mô tả trong bảng sau:

Mô hình/Kịch bản	SPIQA Test - A
	Meteor	Rouge-L	BERTScore-F1	L3Score
Gemini 1.5 Flash	27.1	41.5	69.20	58.12
Gemini 1.5 Pro	27.0	40.4	69.05	64.68
GPT-4 Vision	27.0	39.5	67.24	63.37
GPT-4o	27.4	45.2	69.34	66.09
Kịch bản 1	35.8	26.8	87.67	27.82
Kịch bản 2	26.4	22.6	86.86	11.60
Kịch bản 3	40.9	30.1	88.46	59.56

Thông lượng:
Mô hình/Kịch bản	SPIQA Test-A
	Thông lượng 
Kịch bản 1	10.67 trang/giây
Kịch bản 2	4.56 trang/giây
Kịch bản 3	8.94 trang/giây

