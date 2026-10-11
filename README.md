# Exam Pose Detection

ระบบตรวจจับใบหน้าและประเมินทิศทางศีรษะสำหรับการเฝ้าสังเกตระหว่างการสอบ ใช้ OpenCV, YOLOv8 Face และ 6DRepNet พร้อมติดตามบุคคลข้ามเฟรม แสดงผลผ่านหน้าต่างวิดีโอและเว็บ Dashboard และบันทึกเหตุการณ์ที่เข้าเงื่อนไข `SUSPICIOUS` หรือ `ALERT`

> **สถานะที่ระบบแจ้งเป็นเพียงสัญญาณจากการประเมินทิศทางศีรษะ ไม่ใช่หลักฐานยืนยันการทุจริต**

## ความสามารถหลัก

- ตรวจจับใบหน้าด้วย YOLOv8 Face และติดตามตำแหน่งต่อเนื่องด้วย tracker แบบ BoT-SORT
- กำหนดรหัสติดตามบุคคลในระหว่าง session เพื่อช่วยคง ID เมื่อ tracker เปลี่ยน ID
- ประเมินการหันและก้ม/เงยศีรษะเป็นมุม `Yaw`, `Pitch` และ `Roll` ด้วย 6DRepNet
- ปรับค่าทิศทางอ้างอิงของแต่ละคนจากตัวอย่างช่วงเริ่มต้น แล้วใช้ค่าที่ปรับให้เรียบเพื่อช่วยลดการแกว่งของผลลัพธ์
- แสดงภาพจากกล้องพร้อมกรอบใบหน้า, ID, มุมศีรษะ และสถานะพฤติกรรม
- มีเว็บ Dashboard แสดงภาพล่าสุด จำนวนใบหน้าที่ติดตาม สถานะ และเหตุการณ์ล่าสุด
- บันทึกเหตุการณ์ที่เข้าเงื่อนไข พร้อมภาพ snapshot และข้อมูล JSONL ไว้ในโฟลเดอร์ `logs/`

## สิ่งที่ต้องมี

- Python เวอร์ชัน 3.10 หรือ 3.11
- กล้องเว็บแคม, ไฟล์วิดีโอ หรือ URL สตรีมที่ OpenCV เปิดอ่านได้
- ไฟล์น้ำหนักโมเดล 2 ไฟล์ (ดูหัวข้อ **เตรียมไฟล์โมเดล**)
- แนะนำให้ใช้ NVIDIA GPU ที่รองรับ CUDA เพื่อประมวลผลได้เร็วขึ้น สามารถระบุ `--device cpu` เพื่อใช้ CPU ได้ แต่อาจช้ากว่ามาก

ดาวน์โหลด Python ได้จาก [python.org](https://www.python.org/downloads/) บน Windows ให้เลือกตัวเลือก **Add Python to PATH** ระหว่างติดตั้ง

## ติดตั้ง

1. Clone repository แล้วเข้าโฟลเดอร์โปรเจกต์

   ```bash
   git clone https://github.com/unstaple/Exam-Pose-Detection.git
   cd Exam-Pose-Detection
   ```

2. สร้าง virtual environment

   **Windows (Command Prompt):**
   ```bat
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. ติดตั้งแพ็กเกจที่จำเป็น

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   หากต้องการใช้ GPU ให้ติดตั้ง PyTorch รุ่นที่ตรงกับระบบปฏิบัติการและเวอร์ชัน CUDA ก่อนหรือปรับคำสั่งติดตั้งตาม [คู่มือ PyTorch](https://pytorch.org/get-started/locally/) จากนั้นตรวจสอบว่า PyTorch มองเห็น GPU ได้

## เตรียมไฟล์โมเดล

ไฟล์ในโฟลเดอร์ `Models/` ถูกละเว้นโดย `.gitignore` จึงไม่ได้ดาวน์โหลดมาพร้อม repository ต้องสร้างโฟลเดอร์และวางไฟล์น้ำหนักทั้งสองไฟล์ตามชื่อที่โปรแกรมกำหนด

```text
Exam-Pose-Detection/
└── Models/
    ├── yolov8n-face.pt
    └── 6DRepNet_300W_LP_AFLW2000.pth
```

แหล่งดาวน์โหลด:

- **YOLOv8 Face:** [`yolov8n-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt) — จากโครงการ [akanametov/yolov8-face](https://github.com/akanametov/yolov8-face)
- **6DRepNet:** [`6DRepNet_300W_LP_AFLW2000.pth`](https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth) — ตรวจสอบแหล่งที่มาและเงื่อนไขการใช้งานของโมเดลก่อนนำไปใช้งาน

หากไม่มีไฟล์หรือชื่อไฟล์ไม่ตรงกับด้านบน โปรแกรมจะไม่สามารถเริ่มทำงานได้ โดยสามารถเปลี่ยนตำแหน่งไฟล์ผ่านอาร์กิวเมนต์ `--face-model` และ `--pose-model`

## วิธีรัน

ต้องรันคำสั่งจากโฟลเดอร์หลักของโปรเจกต์ หลังเปิด virtual environment และเตรียมไฟล์โมเดลเรียบร้อยแล้ว

**ใช้กล้องหลักของเครื่อง (ค่าเริ่มต้น):**

```bash
python app.py
```

**ระบุอุปกรณ์ประมวลผล:**

```bash
# NVIDIA GPU ตัวแรก
python app.py --device cuda:0

# CPU
python app.py --device cpu
```

**ใช้ไฟล์วิดีโอแทนกล้อง:**

```bash
python app.py --source "Videos/ชื่อไฟล์วิดีโอ.mp4" --device cuda:0
```

**ใช้ URL ของสตรีมกล้อง (ตัวอย่าง):**

```bash
python app.py --source "http://<IP-ADDRESS>:<PORT>/<STREAM-PATH>" --device cuda:0
```

`--source` รับได้ทั้งหมายเลขกล้อง เช่น `0`, path ของไฟล์วิดีโอ หรือ URL สตรีม หากกล้องไม่ใช่กล้องหมายเลข 0 ให้เปลี่ยนเป็นหมายเลขที่ระบบมองเห็น เช่น `--source 1`

### ตัวเลือกที่ใช้บ่อย

| ตัวเลือก | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `--source` | `0` | หมายเลขกล้อง, path ของวิดีโอ หรือ URL สตรีม |
| `--device` | `cuda:0` | อุปกรณ์ประมวลผล เช่น `cuda:0` หรือ `cpu` |
| `--face-model` | `Models/yolov8n-face.pt` | path ของโมเดลตรวจจับใบหน้า |
| `--pose-model` | `Models/6DRepNet_300W_LP_AFLW2000.pth` | path ของโมเดลประเมินทิศทางศีรษะ |
| `--imgsz` | `1280` | ขนาดภาพที่ส่งให้ YOLO ประมวลผล |
| `--dashboard-host` | `127.0.0.1` | IP ที่ Dashboard ใช้รับการเชื่อมต่อ |
| `--dashboard-port` | `5000` | พอร์ตของ Dashboard |

ตัวอย่างเปลี่ยนพอร์ต Dashboard:

```bash
python app.py --dashboard-port 5001
```

## ใช้งาน Dashboard และหยุดโปรแกรม

เมื่อเริ่มโปรแกรมแล้ว ให้เปิดเบราว์เซอร์ไปที่ [http://127.0.0.1:5000/](http://127.0.0.1:5000/) Dashboard แสดงภาพวิดีโอที่มีการกำกับผล จำนวนใบหน้าที่กำลังติดตาม สถานะพฤติกรรม และรายการเหตุการณ์ล่าสุด

หน้าต่างวิดีโอจาก OpenCV จะแสดงผลแยกต่างหาก กด **`q`** เมื่อหน้าต่างวิดีโออยู่ในโฟกัส หรือกด **Ctrl+C** ใน Terminal เพื่อหยุดโปรแกรม

## ความหมายของสถานะ

- `NORMAL` — ยังไม่เข้าเงื่อนไขการเคลื่อนไหวศีรษะที่กำหนดให้ต้องแจ้งเตือน
- `SUSPICIOUS` — ตรวจพบการหันศีรษะหรือเงยศีรษะเกินค่าเตือนต่อเนื่องตามระยะเวลาที่กำหนด
- `ALERT` — พฤติกรรมที่เข้าเงื่อนไขยังคงอยู่ต่อเนื่องจนถึงระยะเวลาแจ้งเตือน

ค่าตั้งต้นใน `src/behavior_analyzer.py` ได้แก่ มุมหันศีรษะ (`Yaw`) ประมาณ 25° สำหรับระดับเตือนและ 40° สำหรับระดับแจ้งเตือน, มุมเงย (`Pitch`) ประมาณ 15° และ 25° ตามลำดับ รวมถึงระยะเวลาประมาณ 0.8 วินาทีเพื่อเข้าสถานะ `SUSPICIOUS` และ 2.5 วินาทีเพื่อเข้าสถานะ `ALERT` ค่าที่ใช้จริงเป็นมุมเทียบกับ baseline ของแต่ละคนและอาจต้องปรับให้เหมาะกับตำแหน่งกล้อง สภาพห้อง และรูปแบบการสอบ

## ไฟล์บันทึก

เมื่อเริ่มแต่ละ session โปรแกรมจะสร้างโฟลเดอร์ลักษณะนี้โดยอัตโนมัติ:

```text
logs/
└── session_<วันเวลา>_<session-id>/
    ├── events.jsonl
    └── snapshots/
        └── *.jpg
```

- `events.jsonl` เก็บข้อมูลเหตุการณ์ทีละบรรทัด เช่น เวลา, ID ที่ติดตาม, สถานะ, เหตุผล, ระยะเวลา, มุมศีรษะ และตำแหน่งกรอบใบหน้า
- `snapshots/` เก็บภาพที่มีกรอบและข้อความกำกับสำหรับเหตุการณ์ `SUSPICIOUS` หรือ `ALERT`
- โฟลเดอร์ `logs/` ถูกละเว้นจาก Git เพื่อไม่ให้ข้อมูลภายในเครื่องถูก commit โดยไม่ตั้งใจ

## โครงสร้างโปรเจกต์

```text
Exam-Pose-Detection/
├── app.py                       # จุดเริ่มต้นโปรแกรมและวงจรประมวลผลวิดีโอ
├── requirements.txt              # รายการแพ็กเกจ Python
├── config/
│   └── classroom_botsort.yaml    # การตั้งค่า tracker
├── src/
│   ├── tracker.py                # ตรวจจับและติดตามใบหน้า
│   ├── pose_estimator.py         # ประเมินมุมศีรษะด้วย 6DRepNet
│   ├── pipeline.py               # เชื่อมขั้นตอนประมวลผล
│   ├── behavior_analyzer.py      # วิเคราะห์สถานะพฤติกรรม
│   ├── calibration.py             # คำนวณ baseline ของแต่ละคน
│   ├── smoothing.py               # ลดความแกว่งของผลมุม
│   ├── stable_id_manager.py       # จัดการ ID ที่เสถียร
│   ├── logger.py                  # บันทึกเหตุการณ์และภาพ
│   └── dashboard.py               # เว็บ Dashboard และ API
├── templates/                    # HTML ของ Dashboard
├── Videos/                       # ไฟล์วิดีโอตัวอย่าง (ถ้ามี)
├── evaluation_results/            # ผลการประเมิน (ถ้ามี)
├── tests/                         # การทดสอบ
├── Models/                        # ต้องสร้างเองและเพิ่มไฟล์โมเดล
└── logs/                          # สร้างอัตโนมัติเมื่อรันโปรแกรม
```
