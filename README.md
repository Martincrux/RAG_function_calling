**Giới thiệu**

RAG_function_calling là một kho lưu trữ chứa mã để tạo ra các cuộc gọi hàm bằng mô hình Retrieval-Augmented Generation (RAG) tích hợp phương thức function calling. Mô hình này được sử dụng để tạo văn bản tự nhiên bằng cách lấy thông tin từ đoạn văn bản có sẵn.

**Yêu cầu**

Để sử dụng kho lưu trữ này, bạn cần phải có khóa API hợp lệ từ:

* **OpenAI:** https://platform.openai.com/docs/api-reference/introduction
* **Google AI:** https://cloud.google.com/ai-platform/docs/authentication/api-keys

**Hướng dẫn**

Tải kho lưu trữ xuống cục bộ và thêm khóa API của bạn vào tệp `secrets.py` trước khi sử dụng:

```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
GOOGLE_AI_API_KEY = "YOUR_GOOGLE_AI_API_KEY"
```

**Đề xuất**

Để có thời gian xử lý nhanh hơn, chúng tôi khuyên bạn nên chạy mô hình ViSTRAL trên Google Colab bằng cách sử dụng TPU. Bạn có thể tìm thấy hướng dẫn về cách thực hiện việc này tại:

* [Hướng dẫn Google Colab](https://colab.research.google.com/drive/18EnSHo3YkZfht_-0Sx-2exllJhzRuICW?usp=sharing)

Buy me a coffee:
- Momo: 0374525177
- Vietcombank (VCB): 1014622635
