# ==============================================================================
#  This code was written by himmeow the coder.
#  Contact: himmeow.thecoder@gmail.com
#  Discord server: https://discord.gg/deua7trgXc
#
#  Feel free to use and modify this code as you see fit. 
#  If you find it helpful, I'd appreciate a coffee!
#  - Momo: 0374525177
#  - Vietcombank (VCB): 1014622635
# ==============================================================================

from typing import List
from pydantic import BaseModel, Field
import PyPDF2
from scrapegraphai.graphs import PDFScraperGraph
import json

# API Key của Gemini Pro
gemini_key = "..."

# Tạo source
file_path = "employee_info_1.pdf"


# Mở file PDF
with open(file_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Khởi tạo biến sources
    sources = ""

    # Duyệt qua từng trang và nối nội dung vào sources
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        sources += page_text

# # In nội dung của sources
# print(sources)

# Định nghĩa cấu trúc dữ liệu cho lời nhận xét
class Employee(BaseModel):
    name: str = Field(description="Tên của thành viên")
    age: str = Field(description="Tuổi của thành viên")
    role: str = Field(description="Chức vụ của thành viên. Các giá trị hợp lệ bao gồm: 'Giám đốc', 'Thư ký' và 'Nhân viên'")
    hometown: str = Field(description="Quê quán của thành viên (chỉ ghi tên, không ghi đơn vị, ví dụ: 'Quảng Bình', 'Hà Nội'")

# Định nghĩa cấu trúc dữ liệu cho danh sách nhận xét
class Employees(BaseModel):
    employees: List[Employee]

# Cấu hình cho LLM
graph_config = {
    "llm": {
        "api_key":gemini_key,
        "model": "gemini-pro",
    },
}

# Tạo một đối tượng SmartScraperGraph
smart_scraper_graph = PDFScraperGraph(
    prompt="Hãy liệt kê toàn bộ thành viên của công ty này",
    source=sources,
    schema=Employees,  # Định dạng dữ liệu đầu ra
    config=graph_config
)

# Thực thi đồ thị và lưu kết quả vào biến result
result = smart_scraper_graph.run()
print(result)

# Lưu kết quả vào file JSON
result = Employees(**result)
with open('employee_info.json', 'w', encoding='utf-8') as f:
    json.dump(result.dict(), f, indent=4, ensure_ascii=False)
