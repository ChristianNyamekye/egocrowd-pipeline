# iPhone Capture App Specification

## Overview
iOS app that synchronizes iPhone camera, Apple Watch IMU, and UDCAP glove data
into a single timestamped recording session.

## Core Requirements

### 1. Video Recording
- **Camera**: Rear-facing, 4K @ 30fps (configurable to 60fps)
- **LiDAR**: Depth maps captured alongside RGB (iPhone Pro required for depth)
- **ARKit**: Camera pose tracking (6-DoF) at 60Hz
- **Mount**: Chest or head mount (first-person egocentric view)
- **Format**: H.264/HEVC video + separate ARKit pose JSON

### 2. Apple Watch Integration (WatchConnectivity)
- Stream accelerometer + gyroscope data at 100Hz
- Use `CMMotionManager` for raw sensor data
- Also capture `CMDeviceMotion` (sensor-fused attitude)
- Transfer data to iPhone via WatchConnectivity framework
- Sync: use `CMClock` shared timebase between Watch and iPhone

### 3. UDCAP Glove Integration (Bluetooth)
- Connect via BLE (glove broadcasts joint data)
- Receive 21-joint angle data at 120Hz
- Parse using UDCAP SDK protocol (documented in their GitBook)
- Alternative: use their HandDriver app as bridge → WebSocket relay

### 4. Synchronization
- **Master clock**: iPhone `CMClock.hostTimeSyncClock`
- All sensors timestamped relative to master clock
- Sync tolerance: <10ms between any two sensors
- NTP-like sync protocol between Watch ↔ iPhone
- Glove sync via BLE connection event timestamps

### 5. Recording Session Flow
```
[START] → Calibration (5s, hands in neutral pose)
       → Task prompt shown ("Pick up the red mug")
       → Recording (variable length, user controls stop)
       → Review + confirm/discard
       → Upload to cloud
```

### 6. Data Export Format
Per recording session, output:
```
session_YYYYMMDD_HHMMSS/
├── video.mp4              # 4K RGB
├── depth/                 # LiDAR depth maps (16-bit PNG, same FPS as video)
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── arkit_poses.json       # Camera 6-DoF poses [{timestamp, transform_4x4}, ...]
├── watch_imu.json         # IMU data [{timestamp, accel, gyro, attitude}, ...]
├── glove_joints.json      # Joint angles [{timestamp, joints_21}, ...]
├── calibration.json       # Neutral pose, camera intrinsics, sensor offsets
├── metadata.json          # Task description, environment, contributor, kit version
└── sync_log.json          # Clock sync events for post-hoc alignment
```

### 7. Task Prompt System
- Pre-loaded task library (pick/place, open/close, pour, fold, etc.)
- Random task selection with environment-appropriate filtering
- Custom task entry (free text)
- Task difficulty rating (contributor self-report)

### 8. Quality Checks (on-device)
- Video stability check (excessive motion → warning)
- Hand visibility check (hand detection in frame)
- Sensor connectivity check (all 3 sources streaming)
- Minimum duration check (>3 seconds)
- Lighting check (too dark → warning)

## Tech Stack
- **Language**: Swift / SwiftUI
- **Frameworks**: AVFoundation, ARKit, CoreMotion, WatchConnectivity, CoreBluetooth
- **Watch app**: WatchOS companion using CoreMotion
- **Minimum**: iOS 17+ (for latest ARKit), watchOS 10+, iPhone 12 Pro+ (for LiDAR)

## MVP Scope (v0.1)
1. ✅ iPhone video recording with ARKit pose tracking
2. ✅ Apple Watch IMU streaming
3. ⬜ UDCAP glove connection (can be added after, data still valuable without it)
4. ✅ Synchronized export to JSON/video files
5. ✅ Basic task prompt
6. ⬜ Cloud upload (manual AirDrop/Files for MVP)

## Future (v0.2+)
- Automatic cloud upload to S3/GCS
- Contributor onboarding flow
- Gamification (tasks completed, quality scores)
- Remote task assignment from dashboard
- Real-time quality feedback
- Multi-person recording (two hands, two gloves)
