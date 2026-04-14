#ifndef __GESTURE_H__
#define __GESTURE_H__

#include "global.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include "MPU6050.h"

// extern TaskHandle_t GestureTaskHandle;

void gesture_task(void *pvParameters);

#endif