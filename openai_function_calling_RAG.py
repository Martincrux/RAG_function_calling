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

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json, os 

# Cấu hình API key cho OpenAI
os.environ['OPENAI_API_KEY'] = "sk-..."
client = OpenAI()

# Định nghĩa hàm lấy thông tin về nhân viên
def about_employee(info):
    """
    Cung cấp thông tin về các thành viên ban quản lý và nhân viên của công ty
    
    Args:
        info (str): Thông tin cần tìm kiếm về nhân viên
    
    Returns:
        str: Chuỗi JSON chứa thông tin về nhân viên
    """
    
    # Load cơ sở dữ liệu thông tin nhân viên
    db = FAISS.load_local("openai_index\employee_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization = True)
    
    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu
    results = db.similarity_search(info, k=1)  # Lấy kết quả hàng đầu (k=1)
    
    # Chuyển đổi kết quả thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Định nghĩa hàm lấy thông tin về sản phẩm
def about_products(info):
    """
    Cung cấp thông tin về các sản phẩm đang có ở công ty
    
    Args:
        info (str): Thông tin cần tìm kiếm về sản phẩm
    
    Returns:
        str: Chuỗi JSON chứa thông tin về sản phẩm
    """
    
    # Load cơ sở dữ liệu thông tin sản phẩm
    db = FAISS.load_local("openai_index\products_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization = True)
    
    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu
    results = db.similarity_search(info, k=3)  # Lấy 3 kết quả hàng đầu (k=3)
    
    # Chuyển đổi kết quả thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Định nghĩa hàm lấy nhận xét về sản phẩm
def reviews_search(info):
    """
    Cung cấp đánh giá của người dùng về những sản phẩm của công ty
    
    Args:
        info (str): Thông tin cần tìm kiếm về nhận xét
    
    Returns:
        str: Chuỗi JSON chứa các nhận xét về sản phẩm
    """
    
    # Load cơ sở dữ liệu các nhận xét sản phẩm
    db = FAISS.load_local("openai_index\\reviews_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization = True)
    
    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu
    results = db.similarity_search(info, k=5)  # Lấy 5 kết quả hàng đầu (k=5)
    
    # Chuyển đổi kết quả thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Danh sách các hàm có sẵn cho trợ lý ảo
available_functions = {
    "about_employee": about_employee,
    "about_products": about_products,
    "reviews_search": reviews_search
}

# Định nghĩa các công cụ (tools) cho trợ lý ảo
tools = [
    {
        "type": "function",
        "function": {
            "name": "about_employee",
            "description": "Cung cấp những tài liệu liên quan đến người quản lý/nhân viên của công ty mà bạn cần biết.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Thông tin mà bạn cần tìm kiếm, e.g. Email của himmeow the coder.",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "about_products",
            "description": "Cung cấp những tài liệu liên quan đến sản phẩm của công ty mà bạn cần biết.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Thông tin mà bạn cần tìm kiếm, e.g. Xuất xứ của áo phông nam Luôn Vui Tươi.",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reviews_search",
            "description": "Cung cấp những lời nhận xét về các sản phẩm của công ty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Những nhận xét mà bạn cần tìm kiếm, e.g. Nhận xét của người dùng về sản phẩm áo phông nam.",
                    },
                },
                "required": ["info"],
            },
        },
    },
]

# Bộ nhớ của trợ lý ảo, chứa lịch sử trò chuyện
memory = [
    {
        "role": "system", 
        "content": """Bạn là trợ lý ảo thông minh làm việc cho một công ty buôn bán hàng hóa, được cung cấp những công cụ truy xuất dữ liệu để trả lời câu hỏi của người dùng. \n
                     IMPORTANT: LUÔN LUÔN PHẢI tìm thông tin trong các tài liệu bằng tools được cung cấp trước khi trả lời câu hỏi của người dùng!"""
    },
]

# Hàm gửi yêu cầu trò chuyện và xử lý phản hồi
def chat_completion_request(messages, functions=None, model="gpt-4o"):
    """
    Gửi yêu cầu trò chuyện đến OpenAI API và xử lý phản hồi.
    
    Args:
        messages (list): Danh sách tin nhắn trong cuộc trò chuyện.
        functions (list): Danh sách các công cụ có sẵn cho trợ lý ảo.
        model (str): Mô hình ngôn ngữ được sử dụng cho chatbot.
    
    Returns:
        str: Phản hồi từ chatbot hoặc thông báo lỗi.
    """
    
    try:
        # Gửi yêu cầu trò chuyện
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=functions,
            tool_choice="auto", 
            temperature=0,
        )

        # Lấy phản hồi từ OpenAI API
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Nếu có yêu cầu sử dụng công cụ
        if tool_calls:
            # Thêm phản hồi của chatbot vào bộ nhớ
            messages.append(response_message)

            # Xử lý từng yêu cầu sử dụng công cụ
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                
                # Gọi hàm tương ứng với tên công cụ được yêu cầu
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(function_args.get("info"))
                    
                    # Thêm kết quả của công cụ vào bộ nhớ
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                
            # Gửi yêu cầu trò chuyện mới với thông tin bổ sung từ công cụ
            return chat_completion_request(messages=messages, functions=functions)
            
        # Nếu không có yêu cầu sử dụng công cụ, trả về phản hồi trực tiếp
        else:
            msg = response_message.content
            return msg
        
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
# Chạy chatbot
if __name__ == "__main__":
    print("Bắt đầu trò chuyện với trợ lý ảo (nhập 'exit' để dừng)")
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        # Thêm câu hỏi của người dùng vào bộ nhớ
        memory.append({"role": "user", "content": query})
        
        # Gửi yêu cầu trò chuyện và nhận phản hồi
        response = chat_completion_request(messages=memory, functions=tools)
        
        # In phản hồi của chatbot
        print(f"Chatbot: {response}")
        
        # Thêm phản hồi của chatbot vào bộ nhớ
        memory.append({"role": "assistant", "content": response})