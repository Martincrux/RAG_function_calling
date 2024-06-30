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

import os
import json
import re
from typing import Union

# Cấu hình API key cho Google Generative AI
# genai.configure(api_key="...")
# os.environ["GOOGLE_API_KEY"] = "..."

# Định nghĩa hàm lấy thông tin về nhân viên
def search_employee_info_by_name(name: str) -> str:
    """
    Cung cấp những thông tin chi tiết/tài liệu có nhắc đến người có tên giống với bạn cần tìm hiểu, ví dụ như email, số điện thoại, tính cách, sở thích,...

    Args:
        name: Tên đầy đủ của người bạn cần tìm hiểu.
             Ví dụ:
                - "Himmeow", "Nguyễn Văn A"

    Returns:
        Chuỗi JSON chứa tài liệu liên quan về người bạn cần tìm.
    """

    # Load cơ sở dữ liệu thông tin nhân viên từ file
    db = FAISS.load_local(
        "googleai_index/employee_index",
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        allow_dangerous_deserialization=True,
    )

    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu và lấy ra kết quả đầu tiên
    results = db.similarity_search(name, k=5)

    # Chuyển đổi kết quả tìm kiếm thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string


# Định nghĩa hàm tìm kiếm nhân viên bằng các công cụ lọc
def search_employee_by_rules(
    role: str = "pass",
    age: str = "pass",
    hometown: str = "pass",
    sort_by: str = "pass",
    sort_rule: str = "pass",
    choose_range: str = "pass",
) -> str:
    """
    Sử dụng các bộ lọc để cung cấp danh sách và thông tin cơ bản của những người có đặc điểm giống với bạn đang tìm kiếm. Hãy sử dụng tên người phù hợp sau khi thực hiện hàm này để tìm kiếm thông tin chi tiết bằng "search_employee_by_info" nếu cần.
    Đồng thời hàm này cũng cung cấp cho bạn lựa chọn sắp xếp (sort) danh sách nhân viên được lọc theo các tiêu chí, phục vụ cho các truy vấn liên quan đến thứ hạng từ người dùng.
    Lưu ý chỉ sử dụng hàm này khi người dùng yêu cầu tìm kiếm nhân viên dựa trên các thông tin có thể phân loại, bao gồm chức vụ (role), tuổi tác (age) và quê quán (hometown).

    Args:
        role: Danh sách chức vụ hợp lệ mà bạn cần lọc. Các chức vụ hợp lệ bao gồm (Giám đốc, Thư ký, Nhân viên). Đặt role = "pass" nếu bạn không cần sử dụng bộ lọc này.
            Ví dụ:
                - "Giám đốc; Nhân viên".
        age: Danh sách các độ tuổi hợp lệ mà bạn cần lọc. Đặt age = "pass" nếu bạn không cần sử dụng bộ lọc này.
            Ví dụ:
                - "18; 19; 20".
        hometown: Danh sách quê quán hợp lệ mà bạn cần lọc, bao gồm đầy đủ cấp đơn vị hành chính (tỉnh/thành phố) + tên. Đặt hometown = "pass" nếu bạn không sử dụng bộ lọc này.
            Ví dụ:
                - "Tỉnh Quảng Bình; Thành phố Hà Nội".
        sort_by: Tiêu chí sắp xếp mà bạn muốn thực hiện. Các tiêu chí hợp lệ bao gồm (age). Đặt sort_by = "pass" nếu không sử dụng tính năng sắp xếp kết quả tìm kiếm.
            Ví dụ:
                - "age".
        sort_rule: Cách sắp xếp mà bạn muốn áp dụng. Các cách sắp xếp hợp lệ bao gồm (ascending, decrease). Đặt sort_rule = "pass" nếu không dùng tính năng sắp xếp.
            Ví dụ:
                - "ascending"
        choose_range: Ngưỡng thứ hạng mà bạn muốn lấy sau khi đã sắp xếp. Đặt choose_range = "pass" nếu không sử dụng tính năng này.
            Ví dụ:
                - "1; 10"

    Returns:
        Chuỗi JSON chứa danh sách những người phù hợp với yêu cầu của bạn.
    """
    json_file_path = "employee_info.json"
    with open(json_file_path, "r", encoding="utf-8") as f:
        employees = json.load(f)["employees"]

    # Xử lý role
    if role != "pass":
        roles = [r.strip() for r in role.split(";")]
        employees = [e for e in employees if e["role"] in roles]

    # Xử lý age
    if age != "pass":
        ages = [int(a.strip()) for a in age.split(";")]
        employees = [e for e in employees if int(e["age"]) in ages]

    # Xử lý hometown
    if hometown != "pass":
        hometowns = [h.strip().lower() for h in hometown.split(";")]
        employees = [e for e in employees if e["hometown"].lower() in hometowns]

    # Xử lý sort_by
    if sort_by != "pass":
        if sort_by.lower() == "age":
            employees = sorted(
                employees,
                key=lambda x: int(x["age"]),
                reverse=sort_rule.lower() == "decrease",
            )

    # Xử lý choose_range
    if choose_range != "pass":
        start, end = [int(c.strip()) for c in choose_range.split(";")]
        employees = employees[start - 1 : end]

    return json.dumps({"employees": employees}, ensure_ascii=False, indent=4)


# Định nghĩa hàm tìm kiếm nhân viên từ thông tin không thể phân loại
def search_employee_by_info(info: str) -> str:
    """
    Cung cấp danh sách những người có đặc điểm giống với bạn đang tìm kiếm,
    Lưu ý chỉ sử dụng hàm này khi người dùng yêu cầu tìm kiếm nhân viên của công ty
    dựa trên những thông tin không được phân loại và tìm kiếm bằng hàm 'search_employee_by_role'.

    Args:
        info: Đặc điểm/thông tin của người bạn cần tìm.
             Ví dụ:
                - "có email là himmeow.thecoder@gmail.com"

    Returns:
        Chuỗi JSON chứa danh sách những người có thể là người bạn cần tìm và thông tin của họ.
    """

    # Load cơ sở dữ liệu thông tin nhân viên từ file
    db = FAISS.load_local(
        "googleai_index/employee_index",
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        allow_dangerous_deserialization=True,
    )

    # Tìm kiếm thông tin tương đồng trong cơ sở dữ liệu và lấy ra kết quả đầu tiên
    results = db.similarity_search(info, k=5)

    # Chuyển đổi kết quả tìm kiếm thành dạng JSON
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string


# Danh sách các hàm (công cụ) có sẵn cho chatbot
tools = [
    search_employee_info_by_name,
    search_employee_by_rules,
    search_employee_by_info,
]

# Tạo dictionary ánh xạ tên hàm với hàm tương ứng
available_tools = {
    "search_employee_info_by_name": search_employee_info_by_name,
    "search_employee_by_rules": search_employee_by_rules,
    "search_employee_by_info": search_employee_by_info,
}

# Khởi tạo model Google Generative AI với tên model và danh sách công cụ
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=tools,
    system_instruction="""Bạn là một trợ lý ảo thông minh, làm nhiệm vụ quản lý nhân sự cho một công ty. Bạn có khả năng truy xuất thông tin từ cơ sở dữ liệu để trả lời câu hỏi của người dùng một cách chính xác và hiệu quả.
                                                    Hãy nhớ:
                                                        - Luôn luôn sử dụng các công cụ được cung cấp để tìm kiếm thông tin trước khi đưa ra câu trả lời.
                                                        - Trả lời một cách ngắn gọn, dễ hiểu.
                                                        - Không tự ý bịa đặt thông tin.
                                                        - Chú ý tránh lỗi mã hóa đầu vào khi sử dụng các công cụ được cung cấp. LUÔN LUÔN dùng string tiếng việt mã utf 8.
                                                    Lưu ý: Bạn phải sử dụng phối hợp các công cụ được cung cấp để đưa ra câu trả lời cho người dùng, ví dụ, với yêu cầu 'cung cấp email của 3 người lớn tuổi nhất'
                                                    bạn sẽ tìm danh sách những người phù hợp bằng tool 'search_employee_by_rules' sau đấy tìm email của họ bằng 'search_employee_info_by_name'.
                                                """,
)

# Tạo chatbot với system message để cấu hình chatbot
history = []
chat = model.start_chat(history=history)

# Vòng lặp chính để nhận input từ người dùng và trả về phản hồi
while True:
    user_input = input("User: ")

    # Kiểm tra điều kiện thoát
    if user_input.lower() in ["thoát", "exit", "quit"]:
        break

    # Gửi tin nhắn của người dùng cho chatbot và nhận phản hồi
    response = chat.send_message(user_input)

    while True:
        # Tạo dictionary để lưu trữ kết quả từ các hàm
        responses = {}

        # Xử lý từng phần của phản hồi từ chatbot
        for part in response.parts:
            # Kiểm tra xem phần phản hồi có chứa yêu cầu gọi hàm hay không
            if fn := part.function_call:
                function_name = fn.name
                function_args = ", ".join(
                    f"{key}={val}" for key, val in fn.args.items()
                )
                print(function_args)

                # Lấy hàm tương ứng từ available_tools
                function_to_call = available_tools[function_name]

                # Phân tích chuỗi function_args thành dictionary
                function_args_dict = {}
                for item in function_args.split(", "):
                    key, value = item.split("=")
                    function_args_dict[key.strip()] = value.strip()

                # Gọi hàm và lưu kết quả vào dictionary responses
                function_response = function_to_call(**function_args_dict)
                responses[function_name] = function_response

        # Nếu có kết quả từ các hàm
        if responses:
            # Tạo danh sách các phần phản hồi mới, bao gồm kết quả từ các hàm
            response_parts = [
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn, response={"result": val}
                    )
                )
                for fn, val in responses.items()
            ]
            # Gửi phản hồi mới (bao gồm kết quả từ các hàm) cho chatbot
            response = chat.send_message(response_parts)
        else:
            break

    # In ra phản hồi cuối cùng từ chatbot
    print("Chatbot:", response.text)
