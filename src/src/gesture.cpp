#include "gesture.h"

// Handle của task
TaskHandle_t GestureTaskHandle = NULL;

// Định nghĩa hàm task
void gesture_task(void *pvParameters) {
    // 1. Khởi tạo cục bộ bên trong Task
    MPU6050 mpu;
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    Serial.println("Khởi tạo MPU6050 trên Core 1...");
    Wire.begin(11, 12);
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("Lỗi: Không tìm thấy MPU6050! Task sẽ bị hủy.");
        vTaskDelete(NULL); // Xóa task nếu phần cứng lỗi, tránh treo hệ thống
    }

    Serial.println("MPU6050 OK! Bắt đầu thu thập dữ liệu...");

    // 2. Vòng lặp vô hạn của FreeRTOS
    while (1) {
        // Đọc dữ liệu
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        // In ra Serial chuẩn định dạng Edge Impulse
        Serial.printf("%d,%d,%d,%d,%d,%d\n", ax, ay, az, gx, gy, gz);

        // Thay thế delay() của Arduino bằng vTaskDelay của FreeRTOS
        // pdMS_TO_TICKS(20) đổi 20ms thành số tick tương ứng của hệ điều hành
        // 20ms tương đương với tần số lấy mẫu 50Hz
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}