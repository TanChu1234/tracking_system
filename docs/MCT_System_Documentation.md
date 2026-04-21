# 📋 Multi-Camera Tracking (MCT) System Documentation

> **Dự án:** Hệ thống theo dõi đa camera với nhận diện khuôn mặt và tái định danh người (ReID)  
> **Phiên bản:** v2.0 — Multi-Floor Architecture (F1–F7)  
> **Cập nhật:** 2026-02-27

---

## 📑 Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Luồng hoạt động chính](#3-luồng-hoạt-động-chính)
4. [Các module chi tiết](#4-các-module-chi-tiết)
5. [Mô hình AI & Nguyên lý hoạt động](#5-mô-hình-ai--nguyên-lý-hoạt-động)
6. [Cơ sở dữ liệu](#6-cơ-sở-dữ-liệu)
7. [API Server & WebSocket](#7-api-server--websocket)
8. [Hệ thống bản đồ (Image2Map)](#8-hệ-thống-bản-đồ-image2map)
9. [Cấu trúc thư mục](#9-cấu-trúc-thư-mục)
10. [Hướng dẫn triển khai](#10-hướng-dẫn-triển-khai)

---

## 1. Tổng quan hệ thống

### Mục đích
Hệ thống MCT (Multi-Camera Tracking) được xây dựng nhằm **theo dõi và nhận diện nhân viên** di chuyển giữa các tầng và khu vực trong tòa nhà thông qua hệ thống camera giám sát. Hệ thống kết hợp **hai phương pháp nhận diện song song**:

| Phương pháp | Mô hình | Mục đích |
|---|---|---|
| **Face Recognition** (Nhận diện khuôn mặt) | SCRFD + ArcFace | Xác định danh tính (tên/mã NV) khi nhìn thấy khuôn mặt |
| **Body Re-Identification** (Tái định danh qua thân hình) | YOLO11x + TransReID (ViT) | Duy trì ID ổn định khi khuôn mặt bị che hoặc không nhìn thấy |

### Điểm nổi bật
- 🏢 Hỗ trợ **7 tầng** (F1–F7) — mỗi tầng có bản đồ riêng
- 📷 Quản lý **đa camera** từ cơ sở dữ liệu (PostgreSQL)
- 🔄 **Xử lý đa luồng** (ThreadPoolExecutor) — mỗi camera xử lý song song
- 🌐 **WebSocket real-time** — gửi vị trí lên client theo từng tầng
- 🧠 **FAISS Index** (Facebook AI Similarity Search) — tìm kiếm nearest-neighbor với tốc độ cao
- 📦 **Session Management** — quản lý phiên làm việc, ghi log lịch sử di chuyển

---

## 2. Kiến trúc hệ thống

### Sơ đồ tổng thể

```mermaid
flowchart TB
    subgraph Input["📷 Camera Input"]
        CAM1["Camera 1"]
        CAM2["Camera 2"]
        CAM3["Camera 3"]
        CAMN["Camera N"]
    end

    subgraph StreamLayer["🔄 Stream Layer"]
        TS["ThreadedStream\n(Background Thread per Camera)"]
    end

    subgraph Processing["🧠 Frame Processor (per Camera - Multi-threaded)"]
        direction TB
        YOLO["YOLO11x\nPerson Detection"]
        REID["TransReID ViT\nBody ReID Feature Extraction"]
        FACE["SCRFD + ArcFace\nFace Detection & Recognition"]
        YOLO --> REID
        YOLO -.-> FACE
    end

    subgraph IndexLayer["📇 FAISS Indexes"]
        FIDX["Face FAISS Index\n(IndexFlatIP, 512-dim)"]
        RIDX["ReID FAISS Index\n(IndexIDMap, 768-dim)"]
    end

    subgraph TrackingLogic["🔗 Tracking Logic"]
        TA["Track Association\n(IoU Matching)"]
        TV["Temporal ID Voting\n(Majority Vote, 15 frames)"]
        CR["Conflict Resolution\n(Unique ID + Face Name)"]
        TA --> TV --> CR
    end

    subgraph MapSystem["🗺️ Map Manager"]
        HP["Homography Projection\n(Pixel → World mm)"]
        PM["Point Merging\n(Multi-Camera, 130mm threshold)"]
        ROI["ROI Check\n(Desk, Room, Zone)"]
        HP --> PM --> ROI
    end

    subgraph Output["📤 Output"]
        WS["🌐 WebSocket Broadcast\n(Real-time per Floor)"]
        DB["🗄️ PostgreSQL\nSession + Face + Position"]
        API["📡 REST API\n(FastAPI, port 8068)"]
    end

    CAM1 & CAM2 & CAM3 & CAMN -->|RTSP| TS
    TS -->|Frame| Processing
    FACE --> FIDX
    REID --> RIDX
    FIDX & RIDX --> TrackingLogic
    CR --> MapSystem
    ROI --> WS
    ROI --> DB
    API -.->|"Enable/Disable\nFloor & Camera"| Processing
```

### Luồng dữ liệu tổng quan

```mermaid
flowchart LR
    A[📷 Camera RTSP] --> B[ThreadedStream]
    B --> C[FrameProcessor]
    C --> D{Dual Recognition}
    D --> E[🧑 YOLO + TransReID\nBody ReID]
    D --> F[😊 SCRFD + ArcFace\nFace Recognition]
    E --> G[FAISS ReID Index]
    F --> H[FAISS Face Index]
    G --> I[Track Association\n& Conflict Resolution]
    H --> I
    I --> J[Map Manager\nHomography Projection]
    J --> K[🌐 WebSocket\nBroadcast]
    J --> L[🗄️ PostgreSQL\nDB Logging]
```

---

## 3. Luồng hoạt động chính

### 3.1. Quy trình khởi tạo (Startup)

```mermaid
flowchart TD
    A[main_mct.py] --> B[Khởi tạo MCT Session]
    B --> C[Load YOLO11x\nPerson Detection]
    C --> D[Load TransReID\nViT Backbone]
    D --> E[Load Face API\nSCRFD + ArcFace]
    E --> F[Build FAISS Face Index\ntừ thư mục API_Face/faces/]
    F --> G[Khởi tạo Map\ncho 7 tầng F1-F7]
    G --> H{Load Camera Config}
    H -->|DB mode| I[Load từ PostgreSQL]
    H -->|Legacy mode| J[Load từ CLI args]
    I --> K[Register Cameras\nvào Maps từng tầng]
    J --> K
    K --> L[Mở RTSP Streams\nThreadedStream]
    L --> M[Khởi tạo ReIDIndex\n+ FrameProcessor]
    M --> N[Start API Server\nport 8068]
    N --> O[▶️ BẮT ĐẦU MAIN LOOP]
```

### 3.2. Main Processing Loop (Vòng lặp chính)

Mỗi iteration (~30 FPS):

```mermaid
flowchart TD
    A[Kiểm tra Face Directory\nCó ảnh mới?] -->|Có| B[Rebuild FAISS Face Index]
    A -->|Không| C

    B --> C[Đọc frames từ\ntất cả camera streams]
    C --> D[Phân phối xử lý\nđa luồng ThreadPoolExecutor]
    D --> E[Mỗi camera:\nprocess_camera_frame]
    E --> F[Update Maps\n& Broadcast WebSocket]
    F --> G[Sleep 0.03s\n~30 FPS cap]
    G --> A
```

### 3.3. Xử lý từng frame camera (Frame Processing Pipeline)

Đây là **bước quan trọng nhất** — logic cốt lõi của hệ thống:

```mermaid
flowchart TD
    subgraph "STEP A: Face Recognition"
        A1[Phát hiện khuôn mặt\nSCRFD Detector]
        A2[Trích xuất Face Embedding\nArcFace Recognizer]
        A3[Tìm kiếm FAISS Face Index\nso khớp với database]
        A1 --> A2 --> A3
    end

    subgraph "STEP B: Person Detection & ReID"
        B1[Phát hiện người\nYOLO11x class=person]
        B2[Cắt ảnh từng người\n& tính Blur Score]
        B3[Batch Extract ReID Features\nTransReID ViT]
        B4[L2 Normalize Features]
        B1 --> B2 --> B3 --> B4
    end

    subgraph "STEP B.1: Track Association"
        C1[Ghép detection mới\nvới Confirmed Tracks\nbằng IoU]
        C2[ReID Voting\ncho track đã match]
        C3[Temporal ID Voting\nmajority vote over 15 frames]
        C1 --> C2 --> C3
    end

    subgraph "STEP B.2: Conflict Resolution"
        D1[Unique ID Check\n1 ID = 1 person trên camera]
        D2[Face Name Conflict\n1 tên = 1 track duy nhất]
        D1 --> D2
    end

    subgraph "ReID Phase 1 & 2"
        E1[Phase 1: Unmatched Detections\n→ Search FAISS ReID Index]
        E2[Phase 2: Unknown Detections\n→ PendingTrack accumulate\n→ Confirm sau 2 frames]
        E1 --> E2
    end

    A3 --> C1
    B4 --> C1
    C3 --> D1
    D2 --> E1
    E2 --> F[Vẽ Bounding Box\n& Labels lên frame]
```

---

## 4. Các module chi tiết

### 4.1. Package `mct/` — Core Package

```
mct/
├── __init__.py
├── core/                          # Logic cốt lõi
│   ├── engine.py                  # Orchestrator chính — run_demo()
│   ├── frame_processor.py         # FrameProcessor class — xử lý từng frame
│   ├── tracker.py                 # PendingTrack, ConfirmedTrack classes
│   ├── camera_config.py           # Load camera config (DB/CLI), register maps
│   ├── stream.py                  # ThreadedStream — đọc RTSP đa luồng
│   └── map_manager.py             # Chiếu điểm → bản đồ, broadcast WebSocket
├── face/                          # Face Recognition
│   ├── detector.py                # SCRFD + ArcFace + FAISS (setup & detection)
│   └── indexer.py                 # Rebuild FAISS face index khi có ảnh mới
├── reid/                          # Body Re-Identification
│   ├── model.py                   # TransReID ViT model setup
│   ├── features.py                # Feature extraction (single + batch)
│   └── reid_index.py              # ReIDIndex — FAISS index cho Body ReID
└── utils/                         # Tiện ích
    ├── geometry.py                # compute_iou, is_face_inside_body
    ├── file_utils.py              # Monitor face directory (watchdog)
    ├── logging_utils.py           # Floor-based logging, face detection log
    └── time_utils.py              # Time utilities
```

### 4.2. Mô tả vai trò từng file

| File | Vai trò |
|---|---|
| `main_mct.py` | Entry point — parse args, gọi `run_demo()` |
| `engine.py` | Khởi tạo tất cả model, mở streams, chạy main loop |
| `frame_processor.py` | **Logic chính**: Face + YOLO + ReID + Track matching per frame |
| `tracker.py` | Data classes: `PendingTrack` (chờ xác nhận) & `ConfirmedTrack` (đã xác nhận, có temporal voting) |
| `stream.py` | `ThreadedStream` — đọc RTSP stream ở background thread |
| `camera_config.py` | Load camera config từ DB hoặc CLI, register vào Map objects |
| `map_manager.py` | Chiếu pixel → tọa độ thực (mm) trên bản đồ, broadcast WebSocket |
| `detector.py` | Setup SCRFD + ArcFace, `run_face_recognition()` trên 1 frame |
| `indexer.py` | `rebuild_face_index()` — reload khi thư mục faces/ thay đổi |
| `model.py` | `setup_transreid()` — load config + weights cho ViT TransReID |
| `features.py` | `extract_features_batch()` — batch GPU inference cho ReID |
| `reid_index.py` | `ReIDIndex` class — FAISS IndexIDMap + thread-safe operations |
| `api_server.py` | FastAPI + WebSocket — REST API enable/disable + real-time broadcast |
| `database/db_config.py` | Load camera/floor/ROI config từ PostgreSQL |
| `database/mct_tracking.py` | `MCTTracker` — ghi face recognition events + position tracking |
| `image2map/map.py` | `Map` class — camera calibration, homography, point merging |

---

## 5. Mô hình AI & Nguyên lý hoạt động

### 5.1. Person Detection — YOLO11x

```mermaid
flowchart LR
    A["🖼️ Camera Frame\n(BGR image)"] --> B["YOLO11x\nclass=0, person only\nconf=0.5, iou=0.4"]
    B --> C["📦 Bounding Boxes\n(x1, y1, x2, y2) × N người"]
```

- **Model**: YOLO11x (Extra-Large) — `yolo11x.pt` (~114 MB)
- **Chức năng**: Phát hiện tất cả người trong frame
- **Params**: `conf=0.5`, `iou=0.4`
- **Chạy trên GPU** (CUDA)

### 5.2. Body Re-Identification — TransReID (Vision Transformer)

```mermaid
flowchart LR
    A["✂️ Person Crop\n(PIL Image)"] --> B["Transform\nResize → ToTensor\n→ Normalize"]
    B --> C["🧠 TransReID ViT\nVision Transformer\nStride Patch Embedding"]
    C --> D["📊 Feature Vector\n768-dim, L2 normalized"]
```

- **Model**: Vision Transformer (ViT) with TransReID modifications
- **Config**: `configs/Market/vit_transreid_stride.yml`
- **Weights**: `weights/transformer_120.pth` (pretrained trên Market-1501)
- **Backbone**: `jx_vit_base_p16_224` (ViT-Base, patch size 16, input 224×224)
- **Batch processing**: Xử lý tất cả person crops trong 1 frame cùng lúc

### 5.3. Face Recognition — SCRFD + ArcFace

```mermaid
flowchart LR
    A["🖼️ Camera Frame"] --> B["👁️ SCRFD Detector\ndet_10g.onnx"]
    B --> C["📍 Face Boxes\n+ Keypoints"]
    C --> D["🧬 ArcFace Recognizer\nw600k_r50.onnx\nResNet-50"]
    D --> E["📊 Face Embedding\n512-dim, L2 normalized"]
    E --> F["🔍 FAISS IndexFlatIP\nCosine Similarity Search"]
    F --> G{"Similarity > Threshold?"}
    G -->|Có| H["✅ Matched Name\n+ Similarity Score"]
    G -->|Không| I["❌ Unknown"]
```

- **Detector**: SCRFD — model ONNX (`det_10g.onnx`)
- **Recognizer**: ArcFace — model ONNX (`w600k_r50.onnx`, ResNet-50)
- **Database**: Thư mục `API_Face/faces/` chứa ảnh khuôn mặt theo tên
- **Hot-reload**: Monitor thư mục, tự rebuild index khi thêm/xóa ảnh

### 5.4. FAISS Indexes — Nearest Neighbor Search

Hệ thống sử dụng **2 FAISS index** riêng biệt:

#### a) Face FAISS Index
| Thuộc tính | Giá trị |
|---|---|
| **Index Type** | `IndexFlatIP` (Inner Product = Cosine Similarity) |
| **Dimension** | 512 (ArcFace embedding) |
| **Dữ liệu** | Face embeddings từ `API_Face/faces/` |
| **Cập nhật** | Hot-reload khi faces directory thay đổi |

#### b) ReID FAISS Index (`ReIDIndex`)
| Thuộc tính | Giá trị |
|---|---|
| **Index Type** | `IndexIDMap(IndexFlatIP(...))` |
| **Dimension** | 768 (TransReID feature) |
| **Max vectors/person** | 200 |
| **Match Threshold** | 0.7 (cosine similarity) |
| **Thread-safe** | ✅ (multiple locks) |

```mermaid
flowchart TB
    subgraph ReIDIndex["📇 ReIDIndex — Thread-safe FAISS Manager"]
        direction TB
        subgraph FAISS["FAISS IndexIDMap"]
            V0["vector_id=0\nfeat 768d"] --> P0["person_id=0"]
            V1["vector_id=1\nfeat 768d"] --> P0
            V2["vector_id=2\nfeat 768d"] --> P0
            V3["vector_id=3\nfeat 768d"] --> P1["person_id=1"]
            V4["vector_id=4\nfeat 768d"] --> P1
        end
        subgraph Mappings["Mappings"]
            M1["vector_id → person_id"]
            M2["person_id → vector_ids\n(deque, max 200)"]
            M3["person_id → face_name\n(nếu đã nhận diện face)"]
        end
    end

    subgraph Operations["🔧 Operations"]
        S["search(feat)\nTop-K Voting → person_id"] 
        AV["add_vector(feat, pid)\nEvict oldest if > 200"]
        FN["set_face_name(pid, name)\nLiên kết danh tính"]
    end

    FAISS --> Operations
```

**Luồng tìm kiếm ReID (Search Flow):**

```mermaid
flowchart LR
    A["Query Feature\n768-dim, L2 norm"] --> B["FAISS search\nTop-K nearest vectors"]
    B --> C{"Similarity > 0.7?"}
    C -->|Có| D["Vote theo person_id\ncủa vector tìm được"]
    C -->|Không| E["Không match\n→ return -1"]
    D --> F{"Winner có\nmajority vote?"}
    F -->|Có| G["✅ Return person_id\n+ similarity"]
    F -->|Không| E
```

### 5.5. Tracking Logic — Temporal Consistency

#### PendingTrack → ConfirmedTrack

```mermaid
stateDiagram-v2
    [*] --> PendingTrack: Detection mới\nkhông match FAISS
    PendingTrack --> PendingTrack: IoU > 0.5\ntrong frame tiếp theo
    PendingTrack --> ConfirmedTrack: Sau 2 frames\nliên tiếp
    ConfirmedTrack --> ConfirmedTrack: IoU match +\nReID voting
    ConfirmedTrack --> [*]: miss_count > 1000\n(mất dấu)
```

#### Temporal ID Voting (Chống nhảy ID)

**Vấn đề:** ReID search có thể trả về ID khác nhau mỗi frame → ID nhảy liên tục (flickering)

```mermaid
flowchart TD
    subgraph Problem["❌ Vấn đề: ID Flickering"]
        F1["Frame 1: ID=5"] --> F2["Frame 2: ID=7"]
        F2 --> F3["Frame 3: ID=5"]
        F3 --> F4["Frame 4: ID=7"]
        F4 --> F5["... ID nhảy liên tục"]
    end

    subgraph Solution["✅ Giải pháp: Temporal Voting"]
        direction TB
        S1["Mỗi frame:\nThêm ReID result\nvào vote_history"]
        S2["vote_history\ndeque maxlen=15\n[5,5,5,7,5,7,7,7,7,7,7,7,7,7,7]"]
        S3["Đếm votes:\nID=5: 4 phiếu\nID=7: 11 phiếu"]
        S4{"ID=7 chiếm\nmajority > 50%+1?"}
        S5["✅ display_id = 7\n(Chuyển ID ổn định)"]
        S1 --> S2 --> S3 --> S4
        S4 -->|Có| S5
        S4 -->|Không| S6["Giữ display_id cũ"]
    end

    Problem -.->|"Giải quyết bằng"| Solution
```

#### Conflict Resolution

1. **Unique ID per camera**: Nếu 2 track cùng display_id → track có vote_count cao hơn giữ ID, track còn lại bị đổi ID mới
2. **Unique face name per camera**: Nếu 2 track cùng face_name → track có similarity_score cao hơn được gán tên

---

## 6. Cơ sở dữ liệu

### 6.1. Database Schema

Hệ thống sử dụng **PostgreSQL** với 3 bảng chính:

```mermaid
erDiagram
    mct_sessions ||--o{ mct_face_recognition : "session_id"
    mct_sessions ||--o{ mct_position_tracking : "session_id"

    mct_sessions {
        SERIAL id PK
        VARCHAR session_id UK "UUID mỗi lần chạy"
        TIMESTAMP started_at
        TIMESTAMP ended_at
        VARCHAR status "active/stopped/crashed"
        INT total_tracks
        INT total_identified
    }

    mct_face_recognition {
        SERIAL id PK
        VARCHAR session_id FK
        INT local_track_id "display_id"
        VARCHAR usr_id "Mã NV hoặc unknown"
        VARCHAR floor "1F, 3F, 7F..."
        VARCHAR camera_id
        TIMESTAMP detected_at
        FLOAT confidence
    }

    mct_position_tracking {
        SERIAL id PK
        VARCHAR session_id FK
        INT local_track_id
        VARCHAR usr_id
        VARCHAR floor
        FLOAT x "tọa độ X (mm)"
        FLOAT y "tọa độ Y (mm)"
        VARCHAR camera_id
        INT bbox_center_x "pixel X"
        INT bbox_center_y "pixel Y"
        TIMESTAMP tracked_at
    }
```

### 6.2. Sampling Strategy

- **Position Tracking**: Ghi mỗi **5 giây** cho mỗi `local_track_id` (tránh quá tải DB)
- **Face Recognition**: Ghi mỗi khi nhận diện thành công
- **Batched Inserts**: Buffer và flush theo batch để tăng hiệu suất

### 6.3. Truy vấn mẫu

```sql
-- Xem lộ trình di chuyển trong ngày của nhân viên
SELECT floor, x, y, camera_id, tracked_at
FROM mct_position_tracking
WHERE usr_id = 'INF1901002'
  AND DATE(tracked_at) = CURRENT_DATE
ORDER BY tracked_at;

-- Xem tất cả nhận diện khuôn mặt hôm nay
SELECT usr_id, floor, camera_id, detected_at, confidence
FROM mct_face_recognition
WHERE DATE(detected_at) = CURRENT_DATE
  AND usr_id != 'unknown'
ORDER BY detected_at DESC;
```

---

## 7. API Server & WebSocket

### 7.1. REST API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/api/status` | Trạng thái toàn bộ hệ thống |
| `POST` | `/api/floors/{id}/enable` | Bật tracking tầng |
| `POST` | `/api/floors/{id}/disable` | Tắt tracking tầng |
| `POST` | `/api/floors/{id}/toggle` | Đảo trạng thái tầng |
| `POST` | `/api/cameras/{id}/enable` | Bật camera |
| `POST` | `/api/cameras/{id}/disable` | Tắt camera |
| `POST` | `/api/cameras/{id}/toggle` | Đảo trạng thái camera |

### 7.2. WebSocket

```
Kết nối: ws://HOST:8068/ws/{floor_id}

Payload gửi mỗi frame:
{
  "points": [
    {
      "point_id": 5,
      "world_mm": [1234.5, 5678.9],
      "map_px": [120, 340],
      "cameras": ["10.29.98.52"],
      "face_name": "Nguyen Van A"
    }
  ],
  "rois": {
    "desk_1": true,
    "desk_2": false
  },
  "timestamp": "2026-02-27T15:30:00.123456"
}
```

### 7.3. Architecture WebSocket

```
Main Thread                         API Thread (Uvicorn)
───────────                         ────────────────────
map_manager.py                      api_server.py
    │                                    │
    │  send_update(floor, data)          │
    │──────▶ message_queue.put() ──▶ queue_consumer()
    │                                    │
    │                              broadcast() → WebSocket clients
```

---

## 8. Hệ thống bản đồ (Image2Map)

### 8.1. Mục đích
Chiếu tọa độ pixel từ camera → tọa độ thực trên bản đồ tầng (đơn vị **mm**)

### 8.2. Camera Calibration

Mỗi camera cần 2 file calibration:
```
image2map/F{n}/{camera_ip}/
├── intrinsic.yaml   # Ma trận nội tại camera (focal length, optical center)
└── extrinsic.yaml   # Ma trận ngoại tại (rotation, translation) → vị trí camera trong không gian
```

### 8.3. Quy trình chiếu (Homography Projection)

```
Camera Frame (pixel)
        │
        ▼
   Undistort (sử dụng intrinsic params)
        │
        ▼
   Image → World (sử dụng extrinsic matrix)
   (giả sử z_world = -850mm → mặt sàn)
        │
        ▼
   World (mm) → Map (pixel trên bản đồ)
   (sử dụng mm_per_pixel_x = 23.8, mm_per_pixel_y = 23.3)
```

### 8.4. Point Merging

Khi cùng 1 người xuất hiện trên **nhiều camera** cùng tầng:
- Tính khoảng cách giữa các điểm projected
- Nếu khoảng cách < **130mm** → merge thành 1 điểm
- Giữ thông tin từ tất cả cameras

### 8.5. ROI (Region of Interest)

- Mỗi tầng có file `rois.yaml` định nghĩa các khu vực quan tâm (VD: bàn, phòng họp)
- Hệ thống kiểm tra xem person nào đang ở trong ROI nào
- Gửi trạng thái ROI qua WebSocket

---

## 9. Cấu trúc thư mục

```
TransReID/
├── main_mct.py              # 🚀 Entry point
├── api_server.py            # 🌐 FastAPI + WebSocket server
├── init_mct_db.py           # 🗄️ Khởi tạo DB tables
│
├── mct/                     # 📦 Core package
│   ├── core/                #    Orchestration & processing
│   ├── face/                #    Face detection & recognition
│   ├── reid/                #    Body re-identification
│   ├── utils/               #    Utilities
│   └── visualization/       #    (Reserved)
│
├── API_Face/                # 😊 Face Recognition module
│   ├── faces/               #    Ảnh khuôn mặt đã biết (theo tên thư mục)
│   ├── weights/             #    SCRFD + ArcFace ONNX weights
│   ├── service/             #    Processing logic
│   └── load_model.py        #    Model loader
│
├── database/                # 🗄️ Database module
│   ├── db_config.py         #    Load config từ PostgreSQL
│   ├── mct_tracking.py      #    MCTTracker — ghi tracking data
│   └── create_mct_tables.sql #   SQL tạo tables
│
├── image2map/               # 🗺️ Camera-to-Map projection
│   ├── F1/ ... F7/          #    Config cho từng tầng
│   │   ├── {camera_ip}/     #    Calibration files per camera
│   │   │   ├── intrinsic.yaml
│   │   │   └── extrinsic.yaml
│   │   ├── rois.yaml        #    ROI definitions
│   │   └── {n}f.png         #    Map image
│   ├── map.py               #    Map class — homography projection
│   └── camera_calibration.py #   CameraCalibration class
│
├── config/                  # ⚙️ TransReID configuration
│   └── defaults.py          #    YACS config defaults
│
├── configs/                 # 📋 YAML config files
│   └── Market/
│       └── vit_transreid_stride.yml
│
├── model/                   # 🧠 TransReID model architecture
│   ├── make_model.py        #    Model factory
│   └── backbones/           #    ViT backbone definitions
│
├── weights/                 # 🏋️ Pre-trained weights
│   ├── transformer_120.pth  #    TransReID weights
│   └── jx_vit_base_p16_224-80ecf9dd.pth  # ViT pretrained
│
├── yolo11x.pt               # 🔍 YOLO11x person detection weights
│
└── logs/                    # 📝 Runtime logs
```

---

## 10. Hướng dẫn triển khai

### 10.1. Yêu cầu hệ thống

| Yêu cầu | Giá trị |
|---|---|
| **GPU** | NVIDIA GPU với CUDA support (khuyến nghị ≥ 8GB VRAM) |
| **Python** | 3.8+ |
| **PostgreSQL** | Có sẵn database `camera_ai_db` |
| **Network** | Truy cập RTSP stream từ camera |

### 10.2. Cài đặt

```bash
pip install -r requirements.txt
```

Các thư viện chính: `torch`, `torchvision`, `ultralytics`, `faiss-gpu`, `opencv-python`, `fastapi`, `uvicorn`, `psycopg2`, `insightface`, `onnxruntime-gpu`

### 10.3. Chạy hệ thống

```bash
# Mode 1: Load cameras từ database (khuyến nghị)
python main_mct.py --use_db

# Mode 2: Load chỉ vài tầng
python main_mct.py --use_db --floors "3F,1F"

# Mode 3: Legacy — truyền RTSP URL trực tiếp
python main_mct.py --rtsp1 "rtsp://user:pass@ip:port/stream" \
                   --rtsp2 "rtsp://user:pass@ip2:port/stream"
```

### 10.4. Thêm khuôn mặt mới (Hot-reload)

1. Tạo thư mục tên nhân viên trong `API_Face/faces/`, ví dụ: `API_Face/faces/INF1901002_Nguyen_Van_A/`
2. Đặt 1+ ảnh khuôn mặt vào thư mục
3. Hệ thống tự động phát hiện thay đổi → rebuild FAISS index → nhận diện ngay

---

> **Tài liệu được tạo tự động** — Mọi thắc mắc liên hệ team phát triển.
