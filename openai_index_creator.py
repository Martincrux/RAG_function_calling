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

os.environ['OPENAI_API_KEY'] = "sk-..."

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS

# Định nghĩa class kế thừa TextSplitter để chia văn bản theo dòng
class LineTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n')

# Khởi tạo đối tượng LineTextSplitter
text_splitter = LineTextSplitter()

# Đọc dữ liệu từ file reviews.txt
with open('data\\reviews.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Chia văn bản thành các đoạn theo dòng
documents = text_splitter.split_text(text)

db = FAISS.from_texts(documents, OpenAIEmbeddings(model="text-embedding-3-large"))
db.save_local("openai_index\\reviews_index")