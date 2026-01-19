# 📑 ADK Cheatsheet (Android Development Kit)

## 🛠️ Core Tools
- **adb (Android Debug Bridge)**  
  - `adb devices` → List connected devices/emulators  
  - `adb install app.apk` → Install APK on device  
  - `adb uninstall com.package.name` → Uninstall app  
  - `adb logcat` → View system/app logs  
  - `adb shell` → Open device shell  

- **fastboot** (used in bootloader mode)  
  - `fastboot devices` → List devices in fastboot mode  
  - `fastboot flash boot boot.img` → Flash boot image  
  - `fastboot reboot` → Reboot device  

---

## 📱 Device Management
- **Start emulator:**  
  `emulator -avd <name>`  
- **Kill server:**  
  `adb kill-server`  
- **Restart server:**  
  `adb start-server`  
- **Reboot device:**  
  `adb reboot`  

---

## 📂 File Operations
- **Push file to device:**  
  `adb push local.txt /sdcard/remote.txt`  
- **Pull file from device:**  
  `adb pull /sdcard/remote.txt local.txt`  
- **List files:**  
  `adb shell ls /sdcard/`  

---

## 🧪 App Debugging
- **Clear app data:**  
  `adb shell pm clear com.package.name`  
- **Force-stop app:**  
  `adb shell am force-stop com.package.name`  
- **Start activity:**  
  `adb shell am start -n com.package.name/.MainActivity`  

---

## 🔍 Log & Monitoring
- **Logcat filters:**  
  - `adb logcat *:E` → Show only errors  
  - `adb logcat ActivityManager:I *:S` → Filter by tag  
- **Dump system info:**  
  `adb shell dumpsys`  

---

## ⚡ Quick Tips
- Always check device connection: `adb devices`  
- Use `adb root` for root-enabled devices  
- Combine with `grep` for log filtering:  
  `adb logcat | grep "MyApp"`  

---
