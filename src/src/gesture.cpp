#include "gesture.h"
#include "Gesture_ESP32S3_inferencing.h"

float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
int feature_ix = 0; 

void gesture_task(void *pvParameters) 
{
    MPU6050 mpu;
    int16_t ax, ay, az, gx, gy, gz;

    Serial.println("Initializing MPU6050 on Core 1...");
    
    // Initialize I2C bus with previously configured pins
    Wire.begin(11, 12); 
    mpu.initialize();

    if (!mpu.testConnection()) 
    {
        Serial.println("Error: MPU6050 connection failed!");
        vTaskDelete(NULL); 
    }

    Serial.println("MPU6050 OK! Starting AI inference...");

    while (1) 
    {
        if (!mpu.testConnection()) 
        {
            Serial.println("Warning: MPU6050 connection lost! Attempting to reset I2C bus...");
            
            // Completely disable the ESP32 I2C controller
            Wire.end(); 
            
            // Wait 500ms to allow residual voltage on the bus to discharge
            vTaskDelay(pdMS_TO_TICKS(500)); 
            
            // Reinitialize the I2C bus with specific pins
            Wire.begin(11, 12); 
            
            // Wake up and reconfigure the MPU6050
            mpu.initialize();   
            
            // Skip the AI inference below and restart the loop to verify connection
            continue; 
        }
        
        // Read raw accelerometer and gyroscope data
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        // Populate the feature buffer
        features[feature_ix++] = ax;
        features[feature_ix++] = ay;
        features[feature_ix++] = az;
        features[feature_ix++] = gx;
        features[feature_ix++] = gy;
        features[feature_ix++] = gz;

        // Check if the buffer is full (contains enough data for one inference frame)
        if (feature_ix >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) 
        {
            
            // Wrap the raw features into a signal structure compatible with Edge Impulse
            signal_t signal;
            int err = numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
            
            if (err == 0) 
            {
                ei_impulse_result_t result = { 0 };

                // Invoke the Edge Impulse classifier
                EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

                if (res == EI_IMPULSE_OK) 
                {
                    Serial.println("===============================");
                    
                    // Iterate through all trained labels and print their confidence scores
                    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) 
                    {
                        Serial.printf("%s: %.2f\n", result.classification[ix].label, result.classification[ix].value);
                        
                        // Practical logic example: Execute action if confidence exceeds 80%
                        // if (result.classification[ix].value > 0.8) {
                        //     if (strcmp(result.classification[ix].label, "Up_Down") == 0) {
                        //          // Turn on LED logic here
                        //     }
                        // }
                    }
                }
            }
            // Reset the buffer index to start collecting a new frame
            feature_ix = 0;
        }

        // 20ms delay (50Hz sampling rate) to yield CPU to the RTOS scheduler
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