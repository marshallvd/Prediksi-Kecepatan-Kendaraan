# Deteksi Kendaraan dan Estimasi Kecepatan

Repositori ini berisi skrip Python untuk mendeteksi dan melacak kendaraan dalam video menggunakan model YOLO (You Only Look Once) v8 dari Ultralytics. Skrip ini melakukan transformasi perspektif dan anotasi pada video hasilnya.

## Daftar Isi
- [Pendahuluan](#pendahuluan)
- [Dependensi](#dependensi)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Penjelasan Kode](#penjelasan-kode)
  - [Impor Modul dan Definisi Titik Polygon](#impor-modul-dan-definisi-titik-polygon)
  - [Kelas ViewTransformer](#kelas-viewtransformer)
  - [Deteksi dan Pelacakan Kendaraan](#deteksi-dan-pelacakan-kendaraan)
  - [Pengaturan Anotasi](#pengaturan-anotasi)
  - [Proses Frame-by-Frame](#proses-frame-by-frame)
- [Pengaruh Resolusi Video pada Deteksi](#pengaruh-resolusi-video-pada-deteksi)
- [Perhitungan Kecepatan](#perhitungan-kecepatan)
- [Lisensi](#lisensi)

## Pendahuluan
Skrip ini mendeteksi dan melacak kendaraan dalam video, menerapkan transformasi perspektif, dan memberi anotasi pada video dengan kotak pembatas, label, dan perhitungan kecepatan.

## Dependensi
- Python 3.7+
- OpenCV
- NumPy
- Ultralytics YOLO
- Supervision

## Instalasi
1. Clone repositori:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```
2. Instal paket-paket yang diperlukan:
    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan
Jalankan skrip dengan video sumber dan jalur file output:
```bash
python main.py
```
Video Testing dan Hasil:
```bash
https://drive.google.com/drive/folders/10QHum9JRIr6-oki9SrXDmsCdzLbt-c-z?usp=drive_link
```


## Penjelasan Kode

### Impor Modul dan Definisi Titik Polygon
```python
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

SOURCE = np.array([[50, 1100], [1600, 1100], [1700, 200], [1200, 200]])  # CCTV
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)
```
- Berbagai modul diimpor untuk pengolahan video, deteksi objek, transformasi perspektif, dan anotasi.
- `SOURCE` mendefinisikan titik-titik polygon untuk area deteksi dalam video asli.
- `TARGET` mendefinisikan koordinat target untuk transformasi perspektif.

### Kelas ViewTransformer
```python
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
```
- Kelas ini digunakan untuk melakukan transformasi perspektif pada titik-titik yang diberikan, mengubah koordinat dari perspektif asli ke perspektif target.

### Deteksi dan Pelacakan Kendaraan
```python
if __name__ == "__main__":
    source_video_path = "cctv2.mp4"
    target_video_path = "cctv hasil1.mp4"
    confidence_threshold = 0.3
    iou_threshold = 0.7

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    model = YOLO("yolov8x.pt")
```
- `source_video_path` dan `target_video_path`: Lokasi file video sumber dan hasil.
- `confidence_threshold`: Ambang batas kepercayaan untuk deteksi objek.
- `iou_threshold`: Ambang batas Intersection over Union (IoU) untuk Non-Maximum Suppression (NMS).
- `video_info`: Mengambil informasi video seperti resolusi dan frame rate.
- `model`: Memuat model YOLO v8 dari Ultralytics.

### Pengaturan Anotasi
```python
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )
```
- `byte_track`: Inisialisasi pelacak objek ByteTrack.
- `thickness` dan `text_scale`: Menghitung ketebalan garis dan skala teks yang optimal berdasarkan resolusi video.
- `bounding_box_annotator`, `label_annotator`, `trace_annotator`: Pengaturan anotasi bounding box, label, dan jejak.

### Proses Frame-by-Frame
```python
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    vehicle_counts = defaultdict(int)
    counted_vehicles = set()

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            labels = []
            for tracker_id, [_, y], class_id in zip(detections.tracker_id, points, detections.class_id):
                if tracker_id not in counted_vehicles:
                    if class_id == 2:  # Asumsi kelas 2 adalah 'car'
                        vehicle_counts['car'] += 1
                        counted_vehicles.add(tracker_id)
                    elif class_id == 7:  # Asumsi kelas 7 adalah 'truck'
                        vehicle_counts['truck'] += 1
                        counted_vehicles.add(tracker_id)

                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/j")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Gambar zona deteksi
            cv2.polylines(annotated_frame, [SOURCE.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)  # Warna biru untuk zona deteksi

            # Tampilkan jumlah kendaraan di pojok kiri atas dengan warna berbeda
            car_text = f"Cars: {vehicle_counts['car']}"
            truck_text = f"Trucks: {vehicle_counts['truck']}"
            cv2.putText(annotated_frame, car_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness)  # Warna merah untuk mobil
            cv2.putText(annotated_frame, truck_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), thickness)  # Warna hij

au untuk truk

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
```
- Frame diambil dari `frame_generator`.
- Deteksi dilakukan menggunakan model YOLO.
- Deteksi disaring berdasarkan `confidence_threshold` dan zona poligon.
- Pelacakan dilakukan menggunakan ByteTrack.
- Titik-titik deteksi ditransformasikan menggunakan `view_transformer`.
- Kecepatan kendaraan dihitung dan ditampilkan pada frame.

## Pengaruh Resolusi Video pada Deteksi
Resolusi video sangat mempengaruhi hasil deteksi karena:
- **Deteksi Objek**: Model deteksi objek seperti YOLO bekerja lebih baik dengan resolusi yang lebih tinggi karena detail lebih jelas terlihat.
- **Akurasi Bounding Box**: Dengan resolusi tinggi, bounding box akan lebih akurat karena setiap piksel memberikan informasi lebih detail.
- **Perhitungan Kecepatan**: Resolusi yang lebih tinggi memungkinkan pelacakan titik-titik yang lebih akurat, yang penting untuk perhitungan jarak dan kecepatan.

Namun, resolusi yang terlalu tinggi bisa memperlambat pemrosesan karena membutuhkan lebih banyak komputasi.

## Perhitungan Kecepatan
Kecepatan kendaraan dihitung dengan cara berikut:
- **Menghitung Jarak**: Jarak vertikal (sumbu y) yang ditempuh oleh kendaraan dihitung dengan mengambil perbedaan antara posisi awal dan posisi akhir dalam suatu periode waktu.
```python
coordinate_start = coordinates[tracker_id][-1]
coordinate_end = coordinates[tracker_id][0]
distance = abs(coordinate_start - coordinate_end)
```
- **Menghitung Waktu**: Waktu dihitung berdasarkan jumlah frame yang dilalui dalam periode waktu tersebut, dikonversi ke detik.
```python
time = len(coordinates[tracker_id]) / video_info.fps
```
- **Menghitung Kecepatan**: Kecepatan dihitung dengan rumus dasar kecepatan = jarak / waktu dan dikonversi ke km/jam dengan faktor 3.6.
```python
speed = distance / time * 3.6
```

## Lisensi
Proyek ini adalah tugas uas mata kuliah Visi Komputer.
```

Silakan sesuaikan konten ini dengan detail spesifik proyek Anda dan kebutuhan Anda.
