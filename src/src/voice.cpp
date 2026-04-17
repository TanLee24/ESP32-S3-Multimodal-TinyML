#include "voice.h"
#include "VoiceCommand_ESP32-S3_inferencing.h"

// Audio buffer to store 1 second of audio (16000 samples x 2 bytes = 32KB RAM)
int16_t sampleBuffer[VOICE_EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

// Callback function: Automatically invoked by the Edge Impulse SDK to fetch raw audio data and convert it to float
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) 
    {
        out_ptr[i] = (float)sampleBuffer[offset + i];
    }
    return 0;
}

void voice_task(void *pvParameters) 
{
    // 1. Configure Buzzer and LED (Buzzer is Active Low: HIGH is off, LOW is on)
    pinMode(BUZZER_IO, OUTPUT);
    digitalWrite(BUZZER_IO, HIGH);

    pinMode(SINGLE_LED, OUTPUT);

    // 2. Initialize I2S for the INMP441 microphone
    const i2s_config_t i2s_config = 
    {
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

    const i2s_pin_config_t pin_config = 
    {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_start(I2S_PORT);

    Serial.println("Voice Task: Ready to listen for commands!");

    while (1) 
    {
        size_t bytesIn = 0;
        
        // Read a complete 1-second audio block from the I2S microphone
        esp_err_t result = i2s_read(I2S_PORT, &sampleBuffer, sizeof(sampleBuffer), &bytesIn, portMAX_DELAY);

        if (result == ESP_OK && bytesIn > 0) 
        {
            
            // Encapsulate the raw audio signal for AI inference
            signal_t signal;
            signal.total_length = VOICE_EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
            signal.get_data = &microphone_audio_signal_get_data;

            voice_ei_impulse_result_t result_ai = {0};

            // Invoke the voice classification inference engine
            VOICE_EI_IMPULSE_ERROR res = voice_run_classifier(&signal, &result_ai, false);

            if (res == VOICE_EI_IMPULSE_OK) 
            {
                // Iterate through the classification results
                for (size_t ix = 0; ix < VOICE_EI_CLASSIFIER_LABEL_COUNT; ix++) 
                {
                    
                    // Trigger action if confidence score exceeds 80%
                    if (result_ai.classification[ix].value > 0.8) 
                    {
                        
                        // NOTE: Match labels exactly with those trained in Edge Impulse Studio
                        if (strcmp(result_ai.classification[ix].label, "Turn_On") == 0) 
                        {
                            Serial.printf("Command detected: TURN ON (%.2f) -> Long Beep!\n", result_ai.classification[ix].value);
                            digitalWrite(BUZZER_IO, LOW);
                            vTaskDelay(pdMS_TO_TICKS(300));
                            digitalWrite(BUZZER_IO, HIGH);
                            
                            digitalWrite(SINGLE_LED, HIGH);
                        }
                        else if (strcmp(result_ai.classification[ix].label, "Turn_Off") == 0) 
                        {
                            Serial.printf("Command detected: TURN OFF (%.2f) -> Double Beep!\n", result_ai.classification[ix].value);
                            digitalWrite(BUZZER_IO, LOW); vTaskDelay(pdMS_TO_TICKS(100)); digitalWrite(BUZZER_IO, HIGH);
                            vTaskDelay(pdMS_TO_TICKS(100));
                            digitalWrite(BUZZER_IO, LOW); vTaskDelay(pdMS_TO_TICKS(100)); digitalWrite(BUZZER_IO, HIGH);

                            digitalWrite(SINGLE_LED, LOW);
                        }
                    }
                }
            }
        }
        // Brief delay to yield CPU to the RTOS scheduler and background tasks
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}