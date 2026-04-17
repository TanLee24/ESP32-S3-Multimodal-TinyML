#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include <driver/i2s.h>

#define BUZZER_IO GPIO_NUM_4
#define I2S_SD GPIO_NUM_3
#define I2S_SCK GPIO_NUM_2
#define I2S_WS GPIO_NUM_1
#define I2S_PORT I2S_NUM_0
#define SINGLE_LED GPIO_NUM_6 // D3

#endif