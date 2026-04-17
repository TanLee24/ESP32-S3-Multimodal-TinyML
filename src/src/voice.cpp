#include "voice.h"

#include "VoiceCommand_ESP32-S3_inferencing.h"

// Bộ đệm chứa 1 giây âm thanh (16000 mẫu x 2 byte = 32KB RAM)
int16_t sampleBuffer[VOICE_EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

// Hàm Callback: AI sẽ tự động gọi hàm này để lấy dữ liệu âm thanh thô và chuyển sang float
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)sampleBuffer[offset + i];
    }
    return 0;
}

void voice_task(void *pvParameters) {
    // 1. Cấu hình Buzzer (Active Low: Mức CAO là tắt, mức THẤP là kêu)
    pinMode(BUZZER_IO, OUTPUT);
    digitalWrite(BUZZER_IO, HIGH); 

    // 2. Khởi tạo I2S cho INMP441
    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = i2s_bits_per_sample_t(16),
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);

    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_start(I2S_PORT);

    Serial.println("Voice Task: Sẵn sàng nghe lệnh!");

    while (1) {
        size_t bytesIn = 0;
        
        // Đọc nguyên 1 block âm thanh dài 1 giây từ Mic
        esp_err_t result = i2s_read(I2S_PORT, &sampleBuffer, sizeof(sampleBuffer), &bytesIn, portMAX_DELAY);

        if (result == ESP_OK && bytesIn > 0) {
            
            // Đóng gói tín hiệu để nạp vào AI
            signal_t signal;
            signal.total_length = VOICE_EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
            signal.get_data = &microphone_audio_signal_get_data;

            voice_ei_impulse_result_t result_ai = {0};

            // Gọi hàm suy luận (Đã được đổi tên bằng VS Code)
            VOICE_EI_IMPULSE_ERROR res = voice_run_classifier(&signal, &result_ai, false);

            if (res == VOICE_EI_IMPULSE_OK) {
                // Quét kết quả trả về
                for (size_t ix = 0; ix < VOICE_EI_CLASSIFIER_LABEL_COUNT; ix++) {
                    
                    // Nếu phát hiện lệnh với độ tin cậy lớn hơn 90%
                    if (result_ai.classification[ix].value > 0.90) {
                        
                        // LƯU Ý: Thay "Mo_den" / "Tat_den" bằng đúng tên nhãn bạn đã train trên Edge Impulse
                        if (strcmp(result_ai.classification[ix].label, "Turn_On") == 0) {
                            Serial.printf("Nghe lệnh MỞ ĐÈN (%.2f) -> Kêu bíp dài!\n", result_ai.classification[ix].value);
                            digitalWrite(BUZZER_IO, LOW);
                            vTaskDelay(pdMS_TO_TICKS(300));
                            digitalWrite(BUZZER_IO, HIGH);
                        }
                        else if (strcmp(result_ai.classification[ix].label, "Turn_Off") == 0) {
                            Serial.printf("Nghe lệnh TẮT ĐÈN (%.2f) -> Kêu bíp bíp!\n", result_ai.classification[ix].value);
                            digitalWrite(BUZZER_IO, LOW); vTaskDelay(pdMS_TO_TICKS(100)); digitalWrite(BUZZER_IO, HIGH);
                            vTaskDelay(pdMS_TO_TICKS(100));
                            digitalWrite(BUZZER_IO, LOW); vTaskDelay(pdMS_TO_TICKS(100)); digitalWrite(BUZZER_IO, HIGH);
                        }
                    }
                }
            }
        }
        // Tránh chiếm dụng CPU của các tiến trình ngầm
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
}