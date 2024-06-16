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

import os
os.environ["GOOGLE_API_KEY"] = "..."

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS

# Khởi tạo đối tượng LineTextSplitter
class LineTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n')

# Định nghĩa class kế thừa TextSplitter để chia văn bản theo dòng
text_splitter = LineTextSplitter()

# Đọc dữ liệu từ file=
with open('data\\employee_info.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Chia văn bản thành các đoạn theo dòng
documents = text_splitter.split_text(text)

db = FAISS.from_texts(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
db.save_local("googleai_index\\employee_index")