#include "global.h"
#include "led_blinky.h"
#include "gesture.h"

void setup() 
{
    Serial.begin(115200);

    xTaskCreate(ledBlinky, "LED Blinky", 4096, NULL, 1, NULL);
    xTaskCreate(gesture_task, "Gesture Task", 4096, NULL, 1, NULL);
}

void loop() 
{}