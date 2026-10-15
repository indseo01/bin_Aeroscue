# bin_Aeroscue

# Aeroscue

**An AI-powered autonomous drone for search-and-rescue.** Aeroscue detects people and flood in real time, on-device, with no cloud or network required, so it works in disaster zones where infrastructure is gone. Fire and damaged-building detection are planned next.

Built by **Team Arcane** for Smart India Hackathon 2026 (Problem Statement SIH26177, Robotics & Drone, Hardware).

> Status: working detection prototype on Jetson Orin Nano. Flight-controller integration and the dashboard are in progress. See [Roadmap](#roadmap).

---

## What it does

- Runs YOLOv8 object detection on a live RGB camera feed, fully offline
- **Currently detects:** person, flood
- **Planned:** fire, damaged building (see [Roadmap](#roadmap))
- Optimised with TensorRT (FP16) for the Jetson Orin Nano
- Geo-tags detections using GPS + IMU and sends them to a dashboard/command server whenever a connection is available
- If GPS is lost, the IMU estimates position and re-syncs with GPS when it returns

## Hardware used

| Part | Notes |
|---|---|
| NVIDIA Jetson Orin Nano Developer Kit | JetPack 5.1.3 (Python 3.8) |
| Camera | RGB only. We tested with a phone as an IP webcam. A USB or compatible CSI camera also works |
| Flight controller (planned) | Pixhawk-class, via MAVSDK |
| Thermal camera | Not used yet (future work) |

## Repository layout

> TODO: update to match the actual folders.

```
aeroscue/
├── training/        # Colab notebook + dataset merge/remap scripts
├── models/          # trained weights (.pt), ONNX files  (engines are NOT stored, see below)
├── jetson/          # inference + TensorRT conversion scripts
├── dashboard/       # live map dashboard
├── docs/            # SIH presentation, diagrams
└── README.md
```

---

## Quick start

### 1. Train the model (Google Colab)

1. Open `training/train_aeroscue.ipynb` in Colab and set the runtime to **T4 GPU**.
2. Install the pinned version (use the same one on the Jetson):
   ```bash
   pip install ultralytics==8.2.103
   ```
3. Download your datasets and **remap every dataset to the same class IDs** before merging:

   | ID | Class | Status |
   |---|---|---|
   | 0 | person | working |
   | 1 | flood | working |
   | 2 | fire | planned |
   | 3 | damaged_building | planned |

   Keep this ID order fixed. When fire and damaged building are added later, they get the next IDs and nothing needs to be renumbered.

4. Run a 2-epoch test first, then the full training:
   ```python
   from ultralytics import YOLO
   YOLO('yolov8n.pt').train(data='data.yaml', epochs=100, imgsz=960, batch=16,
                            patience=20, project='/content/drive/MyDrive/aeroscue', name='run1')
   ```
   Resume after a disconnect: `YOLO('.../weights/last.pt').train(resume=True)`
5. Export to ONNX:
   ```python
   YOLO('best.pt').export(format='onnx', imgsz=960, opset=12, simplify=True)
   ```

> Do **not** export a TensorRT `.engine` on Colab. Engines only work on the exact GPU and TensorRT version they were built on.

### 2. Set up the Jetson Orin Nano

1. Flash **JetPack 5.1.3** with NVIDIA SDK Manager.
2. Install PyTorch and torchvision. The standard pip wheels do not work on Jetson, so use the NVIDIA wheel for JetPack 5.1 and build a matching torchvision from source. A torch/torchvision version mismatch was our biggest setup problem. Follow NVIDIA's guide: <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>
3. Install the same pinned Ultralytics version:
   ```bash
   pip install ultralytics==8.2.103
   ```
4. TODO: add any other `requirements.txt` dependencies (OpenCV, etc.).

### 3. Build the TensorRT engine (on the Jetson)

```bash
scp best.onnx user@<jetson-ip>:~/aeroscue/models/
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16
```

In our benchmarks (50 warm-up iterations), FP16 gave the biggest speedup, so we use it as the default.

### 4. Run detection

```bash
# TODO: replace with the real script name and arguments
python jetson/detect.py --model models/best_fp16.engine --source <video | image | http://phone-ip:8080/video>
```

If you use a phone as the camera, install an "IP Webcam" app and pass its stream URL as the source.

---

## Known issues and tips

- **Accuracy depends on the data.** Models trained on clean dataset images do poorly on phone or drone footage. Add images that look like your real deployment.
- **Aerial people are tiny.** Train and run at `imgsz` 960 or higher.
- **Class balance matters.** Fire and damaged buildings are scarce in public datasets, so check per-class results.
- **Check label IDs.** A wrong class remap trains without any error and gives bad results. Open a few label files after merging.
- The 15-pin CSI camera ribbon does not fit the Orin Nano's 22-pin connector. Use a compatible module or an adapter cable.

## Roadmap

- [x] Detection pipeline on Jetson Orin Nano
- [x] TensorRT FP16 optimisation
- [x] Person and flood detection
- [ ] Train our own models (replacing the downloaded pretrained ones)
- [ ] Add fire detection
- [ ] Add damaged-building detection
- [ ] Test on real drone video
- [ ] GPS + IMU geo-tagging
- [ ] Live map dashboard (Leaflet.js)
- [ ] Flight-controller integration (MAVSDK / Pixhawk)
- [ ] Thermal camera support (future)
- [ ] SLAM / GPS-denied navigation (future)

## Regulations

Drone flights in India must follow the **Drone Rules 2021** and the DGCA Digital Sky platform. Detection alerts should be verified by a human before action is taken.

## References

- Kunwar et al., "AI-Enhanced UAV Clusters for Search and Rescue in Natural Disasters," *Algorithms*, 2024. <https://doi.org/10.3390/a19010031>
- Speth et al., "Deep Learning with RGB and Thermal Images Onboard a Drone for Monitoring Operations," *Journal of Field Robotics*, 2022. <https://doi.org/10.1002/rob.22101>
- Ultralytics YOLOv8 docs: <https://docs.ultralytics.com/>
- NVIDIA TensorRT docs: <https://docs.nvidia.com/deeplearning/tensorrt/>
- PX4 simulation docs: <https://docs.px4.io/main/en/simulation/>

## Team

Team **Arcane**, Smart India Hackathon 2026 (Team ID 186117).

## License

TODO: choose a license (MIT is a common choice for open projects) and add a `LICENSE` file.
