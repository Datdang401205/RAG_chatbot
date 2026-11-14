# RAG Chatbot - Trợ lý hỏi đáp tài liệu nội bộ

## 1. Tổng quan

RAG Chatbot là một ứng dụng hỏi đáp dựa trên tài liệu (Retrieval-Augmented Generation) cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận câu trả lời được trích xuất từ chính kho tài liệu nội bộ.

Hệ thống được thiết kế theo hướng tối giản, dễ chạy cục bộ, minh bạch nguồn dữ liệu và hạn chế tối đa việc gửi dữ liệu ra bên ngoài:

- **Chat LLM:** Cohere `command-a-03-2025` thông qua `langchain-cohere`.
- **Embedding:** chạy hoàn toàn cục bộ với `BAAI/bge-small-en-v1.5` (384 chiều, đã chuẩn hóa), không gọi API embedding và không cần khóa API.
- **Vector store:** ChromaDB lưu bền trên đĩa tại `chroma_db/`.
- **Giao diện:** Streamlit (`app.py`) và CLI (`src/main.py`).
- **Trích xuất tài liệu:** đọc trực tiếp `.md`/`.txt`; dùng LlamaParse cho `.pdf`/`.docx`/`.pptx`/`.xlsx`, có cache cục bộ.

Mục tiêu chính:

- Trả lời đúng dựa trên ngữ cảnh tài liệu, không tự suy diễn ngoài tài liệu.
- Trả lời `I don't know.` khi thông tin không có trong tài liệu.
- Hiển thị nguồn (tên file) và điểm tương đồng để người dùng kiểm chứng.
- Nạp dữ liệu tăng dần (incremental), chạy lại không tạo bản ghi trùng.

## 2. Kiến trúc

```
Người dùng
   |
   +-- Streamlit UI (app.py) --------+
   |                                 |
   +-- CLI (src/main.py) ------------+
                                     |
                          src/chain.py
                    +----------------+----------------+
                    |                                 |
             Retriever (Chroma)                ChatCohere
                    |                                 |
          HuggingFaceEmbeddings              command-a-03-2025
          BAAI/bge-small-en-v1.5                   |
                    |                          Cohere API
              chroma_db/                            |
                    ^                               v
                    |                       Câu trả lời + nguồn
              src/ingest.py
                    |
        +-----------+-----------+
        |                       |
  papers/*.md,*.txt     LlamaParse (PDF/Office)
                                |
                         .parse_cache/
```

Nguyên tắc thiết kế:

- **Tách biệt embedding và chat:** embedding luôn chạy cục bộ; khóa Cohere chỉ được dùng cho chat.
- **Một factory duy nhất:** `get_llm()` và `get_embeddings()` trong `src/config.py` được dùng lại ở mọi nơi.
- **Không trộn vector:** tên model embedding được lưu trong metadata của collection Chroma; nếu thay đổi model, hệ thống chặn truy vấn và yêu cầu reset.
- **Không dùng RAG của Cohere:** toàn bộ retrieval do Chroma + prompt chuẩn của LangChain đảm nhiệm; không dùng `documents`/connectors của Cohere.

## 3. Tính năng chi tiết

### 3.1. Nạp dữ liệu (`src/ingest.py`)

- Quét đệ quy thư mục `papers/` (biến `DATA_DIR`).
- Hỗ trợ:
  - Đọc trực tiếp: `.md`, `.txt`.
  - Parse qua LlamaParse: `.pdf`, `.docx`, `.pptx`, `.xlsx`.
- Bỏ qua các file có tên bắt đầu bằng `TEST_` (ví dụ `TEST_QUESTIONS.md`).
- Cắt văn bản:
  - Tài liệu parse được chia theo heading markdown (`#`, `##`, `###`) trước.
  - Sau đó tất cả được cắt bằng `RecursiveCharacterTextSplitter` với `chunk_size=800`, `chunk_overlap=100`.
- Sinh ID ổn định cho từng chunk bằng SHA-1 của `source|page|index|content`.
- Chỉ thêm chunk chưa tồn tại; chạy lại lần hai sẽ thêm `0` chunk và báo `All documents already ingested.`
- `--reset`: xóa collection và tạo lại trống trước khi nạp.
- Trả về thống kê: số file, số chunk đã thêm, số file bị bỏ qua, số file parse, số file dùng cache.

Lệnh:

```bash
python -m src.ingest
python -m src.ingest --reset
```

### 3.2. Embedding cục bộ (`src/config.py`)

- Model mặc định: `BAAI/bge-small-en-v1.5`.
- Vector 384 chiều.
- `normalize_embeddings=True` để dùng cosine similarity ổn định.
- Chạy trên CPU, không cần API key, không gửi nội dung tài liệu tới dịch vụ embedding bên ngoài.
- Có thể đổi model qua biến môi trường `EMBED_MODEL`, nhưng phải chạy lại `python -m src.ingest --reset`.

### 3.3. Vector store và bảo vệ model embedding (`src/chain.py`)

- Chroma lưu tại `chroma_db/`, collection tên `rag`.
- Metadata collection lưu `embed_model`.
- Khi khởi tạo vector store:
  - Nếu metadata không khớp `EMBED_MODEL`, hệ thống dừng với thông báo:
    `Embedding model changed: run python -m src.ingest --reset`
  - Điều này ngăn truy vấn bằng vector của model này lên dữ liệu đã embed bằng model khác.
- Hàm `get_vectorstore(check_embedding=False)` được dùng riêng cho thao tác reset.

### 3.4. Truy xuất (`src/chain.py`)

- `similarity_search_with_score()` lấy `k` chunk gần nhất (mặc định `k=4`, có thể chỉnh trên UI).
- Ngữ cảnh được định dạng kèm tên file:

```
[security_policy.txt]
<nội dung chunk>
```

- Nguồn hiển thị trên UI gồm tên file, điểm tương đồng và trích đoạn nội dung.

### 3.5. Sinh câu trả lời và streaming (`src/chain.py`)

- Prompt hệ thống quy định:
  - Chỉ dùng ngữ cảnh được cung cấp.
  - Không dùng kiến thức bên ngoài.
  - Nếu không có thông tin, trả lời chính xác `I don't know.`
  - Trích dẫn tên file nguồn, ví dụ `[security_policy.txt]`.
- Chain dạng chuẩn `prompt | llm` (không dùng RAG/connectors của Cohere).
- Hỗ trợ streaming qua `chain.stream()` và `st.write_stream()`.
- `HISTORY_TURNS=4`: giữ tối đa 4 lượt hội thoại gần nhất (8 message) để trả lời câu hỏi nối tiếp.

### 3.6. Giao diện Streamlit (`app.py`)

- Sidebar:
  - Hiển thị **tên model đang dùng** (không hiển thị khóa API).
  - Nút `Ingest papers/`.
  - Nút `Reset DB + re-ingest`.
  - Thanh chọn `Retriever k` (1-8).
  - Nút `Clear chat`.
  - Danh sách câu hỏi kiểm thử nhanh.
- Khu vực chat:
  - Hiển thị lịch sử hội thoại.
  - Stream câu trả lời theo thời gian thực.
  - Expander `Sources`: tên file, score, trích đoạn.
  - Metadata: độ trễ; token usage nếu response có `usage_metadata`, ngược lại chỉ hiển thị độ trễ.
- Xử lý lỗi:
  - `429`: thử lại tối đa 3 lần với backoff `1s, 2s, 4s`; sau đó báo `Cohere trial rate/usage limit reached`.
  - `401/404`: báo `invalid key or model name`.
  - Lỗi khác: hiển thị thông điệp lỗi API.

Chạy:

```bash
streamlit run app.py
```

### 3.7. CLI (`src/main.py`)

- Vòng lặp hỏi đáp trong terminal.
- Hỗ trợ lịch sử nhiều lượt.
- Streaming từng token ra stdout.
- Gõ `exit` để thoát.

Chạy:

```bash
python -m src.main
```

### 3.8. Kiểm tra khởi động và smoke test

**Startup check** (`src/check.py`):

- Gọi chat một lần với `"hi"` và `max_tokens=5`.
- Embed cục bộ một lần với `"ping"`.
- In một dòng PASS/FAIL cho mỗi bước.
- Dừng ngay nếu chat trả về `401/404`.

Chạy:

```bash
python -m src.check
```

Ví dụ:

```
PASS chat (200, command-a-03-2025, output_tokens=5)
PASS embeddings (200, BAAI/bge-small-en-v1.5, dim=384)
```

**Smoke test** (`scripts/test_smoke.py`):

- Hỏi hai câu kiểm thử: `Who is the CISO?` và `Who is the CTO of Zentora Labs?`.
- Giới hạn số lần gọi Cohere bằng `--max-calls` (mặc định `30`).
- Nghỉ 1 giây giữa các câu hỏi.
- In số lần gọi Cohere thực tế.

Chạy:

```bash
python scripts/test_smoke.py --max-calls 30
```

### 3.9. LlamaParse và cache (`src/parser.py`)

- Dùng `LLAMA_CLOUD_API_KEY` để parse PDF/Office.
- Tier mặc định: `cost_effective`, version `latest`, mở rộng `markdown`.
- Cache tại `.parse_cache/`, tên file là SHA-1 của nội dung file gốc.
- Nếu cache tồn tại, dùng lại và không gọi LlamaParse.
- Nếu thiếu khóa hoặc parse thất bại, file bị bỏ qua với cảnh báo; các file khác vẫn được xử lý.
- Thay đổi nội dung file sẽ tạo cache mới, không ảnh hưởng cache cũ.

## 4. Cấu trúc thư mục

```
RAG_Chatbot/
|-- app.py                  # Giao diện Streamlit
|-- requirements.txt        # Danh sách phụ thuộc
|-- README.md
|-- .env.example            # Mẫu biến môi trường (không chứa khóa thật)
|-- .gitignore
|-- data/
|   `-- .gitkeep
|-- papers/                 # Kho tài liệu nguồn
|   |-- company_handbook.md
|   |-- product_faq.md
|   |-- security_policy.txt
|   |-- sample_report.pdf
|   `-- TEST_QUESTIONS.md   # Bị bỏ qua khi ingest
|-- scripts/
|   `-- test_smoke.py       # Smoke test có giới hạn số lần gọi
`-- src/
    |-- __init__.py
    |-- config.py           # Biến môi trường, factory LLM/embedding
    |-- chain.py            # Vector store, prompt, RAG chain
    |-- ingest.py           # Nạp và cắt tài liệu vào Chroma
    |-- parser.py           # LlamaParse và cache
    |-- main.py             # CLI chat
    `-- check.py            # Startup check
```

## 5. Cài đặt

Yêu cầu: Python 3.10+ (khuyến nghị 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Sau đó điền `COHERE_API_KEY` trong `.env`.

## 6. Cấu hình biến môi trường

| Biến | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `COHERE_API_KEY` | Có (cho chat) | - | Khóa Cohere. Không bắt buộc tiền tố. |
| `COHERE_MODEL` | Không | `command-a-03-2025` | Model chat Cohere. |
| `EMBED_MODEL` | Không | `BAAI/bge-small-en-v1.5` | Model embedding cục bộ. Đổi model phải reset DB. |
| `LLAMA_CLOUD_API_KEY` | Không | - | Khóa LlamaParse cho PDF/Office. Thiếu khóa thì bỏ qua file parse. |

Ghi chú:

- `src/config.py` luôn nạp `.env` bằng đường dẫn tuyệt đối và `load_dotenv(ENV_PATH, override=True)`.
- Ứng dụng đọc `.env`, không đọc `.env.example`.
- Không đặt khóa thật trong `.env.example`.

## 7. Cách sử dụng

```bash
# 1. Nạp tài liệu lần đầu
python -m src.ingest

# 2. Kiểm tra chat + embedding
python -m src.check

# 3. Chạy giao diện web
streamlit run app.py

# 4. Hoặc chạy CLI
python -m src.main

# 5. Kiểm thử nhanh hai câu hỏi
python scripts/test_smoke.py --max-calls 30

# 6. Nạp lại toàn bộ khi đổi EMBED_MODEL hoặc muốn làm sạch DB
python -m src.ingest --reset
```

Quy trình cập nhật tài liệu:

1. Thêm/sửa file trong `papers/`.
2. Chạy `python -m src.ingest`.
3. Chỉ các chunk mới được thêm; chunk cũ không bị nhân đôi.
4. Nếu sửa nội dung file, chunk mới sẽ được thêm; muốn loại bỏ hoàn toàn dữ liệu cũ, chạy `--reset`.

## 8. Mô tả module và hàm

### `src/config.py`

| Hàm / biến | Mô tả |
|---|---|
| `BASE_DIR`, `ENV_PATH` | Thư mục gốc dự án và đường dẫn tuyệt đối tới `.env`. |
| `DATA_DIR`, `PERSIST_DIR` | `papers/` và `chroma_db/`. |
| `COHERE_MODEL`, `EMBED_MODEL` | Tên model chat và embedding. |
| `require_api_key()` | Đọc `COHERE_API_KEY`; lỗi `COHERE_API_KEY missing` nếu rỗng. |
| `get_llm()` | Factory duy nhất tạo `ChatCohere`, temperature `0`. |
| `get_embeddings()` | Factory tạo `HuggingFaceEmbeddings` cục bộ, chuẩn hóa vector. |

### `src/chain.py`

| Hàm | Mô tả |
|---|---|
| `ensure_embed_model(store)` | So sánh `embed_model` trong metadata Chroma với `EMBED_MODEL`; lệch thì raise. |
| `get_vectorstore(check_embedding=True)` | Tạo Chroma với embedding cục bộ và metadata model. |
| `format_docs(docs)` | Ghép chunk thành ngữ cảnh kèm tên file. |
| `build_answer_chain()` | Chain `prompt | llm` cho streaming. |
| `build_rag_chain()` | Chain đầy đủ retrieval + prompt + LLM + parser. |

### `src/ingest.py`

| Hàm | Mô tả |
|---|---|
| `_list_files()` | Liệt kê file hợp lệ trong `papers/`, bỏ file `TEST_*`. |
| `_load(path)` | Đọc trực tiếp file text/markdown. |
| `_chunk_id(chunk, index)` | Sinh ID SHA-1 ổn định cho chunk. |
| `_split(docs, markdown)` | Chia heading markdown (nếu là tài liệu parse) và cắt chunk. |
| `ingest(reset=False)` | Nạp tăng dần vào Chroma, trả về thống kê. |

### `src/parser.py`

| Hàm | Mô tả |
|---|---|
| `cache_path(path)` | Đường dẫn cache dựa trên SHA-1 nội dung file. |
| `parse_to_documents(path)` | Đọc cache hoặc gọi LlamaParse, trả về danh sách `Document`. |

### `app.py`

| Hàm | Mô tả |
|---|---|
| `get_resources(k)` | Cache vector store, answer chain và trạng thái có dữ liệu. |
| `_status_code(exc)` | Lấy HTTP status từ exception. |
| `_is_429(exc)` | Nhận diện lỗi rate limit. |
| `render_meta(sources, latency, usage)` | Hiển thị nguồn, độ trễ, token usage. |

### `src/check.py`

- `main()`: chạy một chat call ngắn và một embedding call, in PASS/FAIL.

### `scripts/test_smoke.py`

- `main()`: chạy hai câu hỏi kiểm thử, tuân thủ `--max-calls`, nghỉ 1 giây giữa các câu.

## 9. Luồng xử lý một câu hỏi

1. Người dùng nhập câu hỏi.
2. Hệ thống embed câu hỏi bằng model cục bộ.
3. Chroma tìm `k` chunk gần nhất.
4. Chunk được ghép thành ngữ cảnh kèm tên file.
5. Prompt hệ thống + lịch sử hội thoại + ngữ cảnh + câu hỏi được gửi tới Cohere.
6. Câu trả lời được stream về UI/CLI.
7. UI hiển thị nguồn, score, độ trễ và token usage (nếu có).

## 10. Xử lý lỗi và giới hạn

| Tình huống | Hành vi |
|---|---|
| `COHERE_API_KEY` rỗng | Báo `COHERE_API_KEY missing`. |
| Cohere `429` | Retry 3 lần với backoff 1s/2s/4s; sau đó báo `Cohere trial rate/usage limit reached`. |
| Cohere `401/404` | Báo `invalid key or model name`. |
| Đổi `EMBED_MODEL` | Chặn truy vấn, yêu cầu `python -m src.ingest --reset`. |
| Thiếu `LLAMA_CLOUD_API_KEY` | Bỏ qua file PDF/Office, vẫn xử lý file text/markdown. |
| Vector DB rỗng | UI cảnh báo và yêu cầu ingest. |
| Chạy ingest lần hai | Thêm 0 chunk, báo tất cả đã được nạp. |

## 11. Kiểm thử và kết quả mẫu

Bộ câu hỏi đầy đủ nằm trong `papers/TEST_QUESTIONS.md`.

Ví dụ kiểm thử:

```bash
python -m src.ingest --reset
python -m src.ingest          # lần hai: Added 0 chunks
python -m src.check
python scripts/test_smoke.py --max-calls 30
```

Kết quả kỳ vọng:

- `Who is the CISO?` -> Tomas Lindqvist, trích dẫn `security_policy.txt`.
- `Who is the CTO of Zentora Labs?` -> `I don't know.`
- Chat trả về đúng model `command-a-03-2025`, embedding 384 chiều.
- Smoke test dùng 2 lần gọi Cohere, dưới giới hạn 30.

## 12. Bảo mật

- `.env` nằm trong `.gitignore`, không được commit.
- `.env.example` chỉ chứa giá trị mẫu/placeholder.
- Không ghi log, không hiển thị khóa API trên UI hoặc CLI.
- Embedding chạy cục bộ nên nội dung tài liệu không bị gửi tới nhà cung cấp embedding.
- Chỉ nội dung ngữ cảnh cần thiết được gửi tới Cohere cho bước sinh câu trả lời.
- Khi chia sẻ repo, kiểm tra lại không có khóa thật trong file được track.

## 13. Hạn chế và hướng phát triển

- Chưa có xác thực người dùng cho giao diện Streamlit.
- Chưa hỗ trợ xóa riêng lẻ theo từng tài liệu; hiện chỉ có nạp tăng dần hoặc reset toàn bộ.
- Chưa có reranker; chất lượng phụ thuộc vào embedding + top-k.
- Chưa có đánh giá tự động (offline evaluation) cho độ chính xác retrieval/answer.
- Có thể bổ sung:
  - Rerank để tăng độ chính xác.
  - Lưu lịch sử hội thoại theo phiên.
  - Theo dõi chi phí/token theo người dùng.
  - Hỗ trợ thêm định dạng tài liệu và pipeline OCR nâng cao.
