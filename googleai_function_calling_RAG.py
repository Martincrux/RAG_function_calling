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

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import os, json

# Cấu hình API key cho Google Generative AI
genai.configure(api_key="...")
os.environ["GOOGLE_API_KEY"] = "..."

# Định nghĩa hàm lấy thông tin về nhân viên
def about_employee(info: str) -> str:
    """
    Cung cấp thông tin về các thành viên ban quản lý và nhân viên của công ty.

    Hàm này tìm kiếm thông tin về nhân viên trong cơ sở dữ liệu dựa trên thông tin đầu vào.

    Args:
        info: Thông tin cần tìm kiếm về nhân viên. 
             Ví dụ: 
                - "Email của himmeow the coder"

    Returns:
        Chuỗi JSON chứa thông tin về nhân viên tìm thấy, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    # Load cơ sở dữ liệu thông tin nhân viên từ file
    db = FAISS.load_local("googleai_index\employee_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    
    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu và lấy ra kết quả đầu tiên
    results = db.similarity_search(info, k=1)
    
    # Chuyển đổi kết quả tìm kiếm thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Định nghĩa hàm lấy thông tin về sản phẩm
def about_products(info: str) -> str:
    """
    Cung cấp thông tin về các sản phẩm đang có ở công ty.

    Hàm này tìm kiếm thông tin về sản phẩm trong cơ sở dữ liệu dựa trên thông tin đầu vào.

    Args:
        info: Thông tin cần tìm kiếm về sản phẩm.
             Ví dụ:
                - "Điện thoại thông minh"

    Returns:
        Chuỗi JSON chứa thông tin về tối đa 3 sản phẩm tìm thấy, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    # Load cơ sở dữ liệu thông tin sản phẩm từ file
    db = FAISS.load_local("googleai_index\products_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    
    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu và lấy ra tối đa 3 kết quả
    results = db.similarity_search(info, k=3)
    
    # Chuyển đổi kết quả tìm kiếm thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Định nghĩa hàm lấy đánh giá về sản phẩm
def reviews_search(info: str) -> str:
    """
    Cung cấp đánh giá của người dùng về những sản phẩm của công ty.

    Hàm này tìm kiếm đánh giá của người dùng về sản phẩm dựa trên thông tin đầu vào.

    Args:
        info: Thông tin cần tìm kiếm trong đánh giá. 
             Ví dụ:
                - "Nhận xét về áo phông nam"

    Returns:
        Chuỗi JSON chứa tối đa 5 đánh giá tìm thấy, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    # Load cơ sở dữ liệu đánh giá sản phẩm từ file
    db = FAISS.load_local("googleai_index\\reviews_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    
    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu và lấy ra tối đa 5 kết quả
    results = db.similarity_search(info, k=5)
    
    # Chuyển đổi kết quả tìm kiếm thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Danh sách các hàm (công cụ) có sẵn cho chatbot
tools = [about_employee, about_products, reviews_search]

# Tạo dictionary ánh xạ tên hàm với hàm tương ứng
available_tools = {
    "about_employee": about_employee,
    "about_products": about_products,
    "reviews_search": reviews_search
}

# Khởi tạo model Google Generative AI với tên model và danh sách công cụ
model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=tools)

# Tạo chatbot với system message để cấu hình chatbot

history=[
    {
      "role": "user",
      "parts": [
        """Bạn là một trợ lý ảo thông minh, làm việc cho một công ty bán hàng hóa. Bạn có khả năng truy xuất thông tin từ cơ sở dữ liệu để trả lời câu hỏi của người dùng một cách chính xác và hiệu quả.

        Hãy nhớ: 
            - Luôn luôn sử dụng các công cụ được cung cấp để tìm kiếm thông tin trước khi đưa ra câu trả lời.
            - Trả lời một cách ngắn gọn, dễ hiểu.
            - Không tự ý bịa đặt thông tin.
        """,
      ],
    },
]

chat = model.start_chat(history=history)


# Vòng lặp chính để nhận input từ người dùng và trả về phản hồi
while True:
    user_input = input("User: ")
    
    # Kiểm tra điều kiện thoát
    if user_input.lower() in ["thoát", "exit", "quit"]:
        break
    
    # Gửi tin nhắn của người dùng cho chatbot và nhận phản hồi
    response = chat.send_message(user_input)

    # Tạo dictionary để lưu trữ kết quả từ các hàm
    responses = {}

    # Xử lý từng phần của phản hồi từ chatbot
    for part in response.parts:
        # Kiểm tra xem phần phản hồi có chứa yêu cầu gọi hàm hay không
        if fn := part.function_call:
            function_name = fn.name
            function_args = ", ".join(f"{key}={val}" for key, val in fn.args.items())

            # Lấy hàm tương ứng từ available_tools
            function_to_call = available_tools[function_name]
            
            # Gọi hàm và lưu kết quả vào dictionary responses
            function_response = function_to_call(function_args)
            responses[function_name] = function_response

            # print(responses) # In ra kết quả từ các hàm (có thể comment dòng này)

    # Nếu có kết quả từ các hàm
    if responses:
        # Tạo danh sách các phần phản hồi mới, bao gồm kết quả từ các hàm
        response_parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
            for fn, val in responses.items()
        ]
        # Gửi phản hồi mới (bao gồm kết quả từ các hàm) cho chatbot
        response = chat.send_message(response_parts)

    # In ra phản hồi cuối cùng từ chatbot
    print("Chatbot:", response.text)