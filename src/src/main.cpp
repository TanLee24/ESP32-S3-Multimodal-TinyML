#include "global.h"
#include "led_blinky.h"
#include "gesture.h"
#include "voice.h"

void setup() 
{
    Serial.begin(115200);

    xTaskCreatePinnedToCore(ledBlinky, "LED Blinky", 4096, NULL, 1, NULL, 1); // Run on Core 1
    xTaskCreatePinnedToCore(voice_task, "Voice Task", 16384, NULL, 1, NULL, 0); // Run on Core 0
    // xTaskCreatePinnedToCore(gesture_task, "Gesture Task", 16384, NULL, 1, NULL, 1); // Run on Core 1
}

void loop() 
{
    vTaskDelete(NULL);
}