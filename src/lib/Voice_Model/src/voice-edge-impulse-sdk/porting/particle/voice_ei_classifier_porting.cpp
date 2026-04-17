/* The Clear BSD License
 *
 * Copyright (c) 2025 EdgeImpulse Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "../voice_ei_classifier_porting.h"
#if VOICE_EI_PORTING_PARTICLE == 1

#include <Particle.h>
#include <stdarg.h>
#include <stdlib.h>

#define VOICE_EI_WEAK_FN __attribute__((weak))

VOICE_EI_WEAK_FN VOICE_EI_IMPULSE_ERROR voice_ei_run_impulse_check_canceled() {
    return VOICE_EI_IMPULSE_OK;
}

VOICE_EI_WEAK_FN VOICE_EI_IMPULSE_ERROR voice_ei_sleep(int32_t time_ms) {
    delay(time_ms);
    return VOICE_EI_IMPULSE_OK;
}

uint64_t voice_ei_read_timer_ms() {
    return millis();
}

uint64_t voice_ei_read_timer_us() {
    return micros();
}

void voice_ei_serial_set_baudrate(int baudrate)
{

}

VOICE_EI_WEAK_FN void voice_ei_putchar(char c)
{
    Serial.write(c);
}

VOICE_EI_WEAK_FN char voice_ei_getchar()
{
    char ch = 0;
    if (Serial.available() > 0) {
	    ch = Serial.read();
    }
    return ch;
}

/**
 *  Printf function uses vsnprintf and output using Arduino Serial
 */
__attribute__((weak)) void voice_ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}

__attribute__((weak)) void voice_ei_printf_float(float f) {
    Serial.print(f, 6);
}

__attribute__((weak)) void *voice_ei_malloc(size_t size) {
    return malloc(size);
}

__attribute__((weak)) void *voice_ei_calloc(size_t nitems, size_t size) {
    return calloc(nitems, size);
}

__attribute__((weak)) void voice_ei_free(void *ptr) {
    free(ptr);
}

#if defined(__cplusplus) && VOICE_EI_C_LINKAGE == 1
extern "C"
#endif
__attribute__((weak)) void DebugLog(const char* s) {
    voice_ei_printf("%s", s);
}

#endif // VOICE_EI_PORTING_PARTICLE == 1
