#include "led_blinky.h"

void ledBlinky(void *pvParameters)
{
    pinMode(LED_PIN, OUTPUT);
    
    while (1)
    {
        digitalWrite(LED_PIN, HIGH);
        vTaskDelay(1000);
        digitalWrite(LED_PIN, LOW);
        vTaskDelay(1000);
    }
    
}