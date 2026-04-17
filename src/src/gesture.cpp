#include "gesture.h"

#include "Gesture_ESP32S3_inferencing.h"

float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
int feature_ix = 0; 

void gesture_task(void *pvParameters) {
    MPU6050 mpu;
    int16_t ax, ay, az, gx, gy, gz;

    Serial.println("Khởi tạo MPU6050 trên Core 1...");
    // Dùng đúng chân I2C bạn đã cấu hình thành công lúc trước
    Wire.begin(11, 12); 
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("Lỗi: Không tìm thấy MPU6050!");
        vTaskDelete(NULL); 
    }

    Serial.println("MPU6050 OK! Bắt đầu nhận diện AI...");

    while (1) {
        if (!mpu.testConnection()) {
            Serial.println("Cảnh báo: Mất kết nối MPU6050! Đang thử reset bus I2C...");
            
            // Tắt hẳn bộ điều khiển I2C của ESP32
            Wire.end(); 
            
            // Đợi nửa giây cho điện áp trên dây và cảm biến xả hết
            vTaskDelay(pdMS_TO_TICKS(500)); 
            
            // Khởi tạo lại đường truyền (Sử dụng đúng chân bạn đang cắm)
            Wire.begin(11, 12); 
            
            // Đánh thức và cấu hình lại MPU6050
            mpu.initialize();   
            
            // Bỏ qua phần xử lý AI bên dưới, quay lại đầu vòng lặp để check tiếp
            continue; 
        }
        
        // Đọc dữ liệu từ cảm biến
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        // Nạp dữ liệu vào mảng (bộ đệm)
        features[feature_ix++] = ax;
        features[feature_ix++] = ay;
        features[feature_ix++] = az;
        features[feature_ix++] = gx;
        features[feature_ix++] = gy;
        features[feature_ix++] = gz;

        // 3. Nếu mảng đã đầy (đủ dữ liệu cho 1 lần nhận diện)
        if (feature_ix >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
            
            // Đóng gói mảng features thành cấu trúc signal mà AI hiểu
            signal_t signal;
            int err = numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
            
            if (err == 0) {
                ei_impulse_result_t result = { 0 };

                // Gọi hàm suy luận của AI (Phép màu nằm ở đây)
                EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

                if (res == EI_IMPULSE_OK) {
                    Serial.println("===============================");
                    // Quét qua các nhãn (Labels) và in ra % độ chính xác
                    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                        Serial.printf("%s: %.2f\n", result.classification[ix].label, result.classification[ix].value);
                        
                        // Ví dụ logic thực tế: Nếu nhận diện > 80% thì làm gì đó
                        // if (result.classification[ix].value > 0.8) {
                        //     if (strcmp(result.classification[ix].label, "Up_Down") == 0) {
                        //          // Code bật LED ở đây
                        //     }
                        // }
                    }
                }
            }
            // Reset biến đếm để bắt đầu thu thập vòng mới
            feature_ix = 0;
        }

        // Delay 20ms (tần số 50Hz) để nhường CPU cho hệ điều hành
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}



// void gesture_task(void *pvParameters) {
//     MPU6050 mpu;
//     int16_t ax, ay, az, gx, gy, gz;

//     Serial.println("Khởi tạo MPU6050 để THU THẬP DỮ LIỆU...");
//     Wire.begin(11, 12); // Chân SDA, SCL của bạn
//     mpu.initialize();

//     if (!mpu.testConnection()) {
//         Serial.println("Lỗi: Không tìm thấy MPU6050!");
//         vTaskDelete(NULL); 
//     }

//     // Đợi 2 giây trước khi bắt đầu in số
//     vTaskDelay(pdMS_TO_TICKS(2000));

//     while (1) {
//         // Đọc dữ liệu
//         mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

//         // In ra Serial theo định dạng chuẩn của Edge Impulse
//         Serial.printf("%d,%d,%d,%d,%d,%d\n", ax, ay, az, gx, gy, gz);

//         // Delay 20ms (Tần số 50Hz)
//         vTaskDelay(pdMS_TO_TICKS(20));
//     }
// }