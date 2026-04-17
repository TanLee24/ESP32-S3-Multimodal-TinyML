import os

replacements = {
    # 1. Các tiền tố tiêu chuẩn
    "ei_": "voice_ei_",
    "EI_": "VOICE_EI_",
    "edge-impulse-sdk": "voice-edge-impulse-sdk",
    "edge_impulse": "voice_edge_impulse",
    "EDGE_IMPULSE": "VOICE_EDGE_IMPULSE",
    "model-parameters": "voice-model-parameters",
    "tflite-model": "voice-tflite-model",
    "tensorflow-lite/tensorflow/": "voice-edge-impulse-sdk/tensorflow/",
    "run_classifier": "voice_run_classifier",
    
    # 2. Lưới vét các hàm viết ẩu của Edge Impulse (Trị dứt điểm lỗi Linker)
    "init_postprocessing": "voice_init_postprocessing",
    "deinit_postprocessing": "voice_deinit_postprocessing",
    "run_postprocessing": "voice_run_postprocessing",
    "display_postprocessing": "voice_display_postprocessing",
    "run_inference": "voice_run_inference",
    "init_impulse": "voice_init_impulse",
    "process_impulse": "voice_process_impulse",
    "flatten_class": "voice_flatten_class"
}

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith((".c", ".cpp", ".cc", ".h", ".hpp", ".S", ".txt")):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            for old, new in replacements.items():
                content = content.replace(old, new)
                
            with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
                f.write(content)

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        new_name = name
        for old, new in replacements.items():
            if "/" in old: continue
            new_name = new_name.replace(old, new)
        if new_name != name:
            os.rename(os.path.join(root, name), os.path.join(root, new_name))
    
    for name in dirs:
        new_name = name
        for old, new in replacements.items():
            if "/" in old: continue
            new_name = new_name.replace(old, new)
        if new_name != name:
            os.rename(os.path.join(root, name), os.path.join(root, new_name))

print("Script V3 đã dọn sạch toàn bộ hàm ẩn! Sẵn sàng liên kết.")