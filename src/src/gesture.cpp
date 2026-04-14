#include "gesture.h"

#include "Gesture_ESP32S3_inferencing.h"

// Các biến cho AI
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static uint32_t feature_ix = 0;

void gesture_task(void *pvParameters) {
    MPU6050 mpu;
    int16_t ax, ay, az, gx, gy, gz;

    Wire.begin(21, 22);
    mpu.initialize();
    if (!mpu.testConnection()) {
        vTaskDelete(NULL);
    }

    while (1) {
        // 1. Đọc dữ liệu thô
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        // 2. Đưa vào mảng features (Edge Impulse cần đơn vị m/s^2 và deg/s nếu bạn train như vậy, 
        // nhưng Data Forwarder truyền số thô nên ta cứ dùng số thô nếu lúc train dùng số thô)
        features[feature_ix++] = (float)ax;
        features[feature_ix++] = (float)ay;
        features[feature_ix++] = (float)az;
        features[feature_ix++] = (float)gx;
        features[feature_ix++] = (float)gy;
        features[feature_ix++] = (float)gz;

        // 3. Khi đã đủ dữ liệu cho 1 cửa sổ (window)
        if (feature_ix >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
            signal_t signal;
            numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

            ei_impulse_result_t result = { 0 };
            // Chạy AI suy luận
            EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
            
            if (res == EI_IMPULSE_OK) {
                // In kết quả dự đoán
                for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                    if (result.classification[ix].value > 0.8) {
                        Serial.printf("Cử chỉ phát hiện: %s (Tin cậy: %.2f)\n", 
                                      result.classification[ix].label, 
                                      result.classification[ix].value);
                        
                        // Thực hiện logic điều khiển LED ở đây
                    }
                }
            }

            // Reset index để thu thập cửa sổ tiếp theo
            feature_ix = 0;
        }

        // Tần số lấy mẫu phải khớp với lúc train (ví dụ 50Hz = 20ms)
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}