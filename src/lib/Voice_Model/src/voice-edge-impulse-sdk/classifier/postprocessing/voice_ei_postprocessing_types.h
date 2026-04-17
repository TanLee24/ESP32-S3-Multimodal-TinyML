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
 *//* The Clear BSD License
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

#ifndef VOICE_EI_POSTPROCESSING_TYPES_H
#define VOICE_EI_POSTPROCESSING_TYPES_H

#include <functional>
#include <string>
#include <type_traits>
#include "voice-edge-impulse-sdk/classifier/voice_ei_model_types.h"

typedef struct cube {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    float confidence;
    const char *label;
} voice_ei_classifier_cube_t;

typedef struct {
    float threshold;
} voice_ei_fill_result_object_detection_threshold_config_t;

typedef struct {
    float threshold;
    uint8_t version;
    uint32_t object_detection_count;
    uint32_t output_features_count;
    voice_ei_object_detection_nms_config_t nms_config;
} voice_ei_fill_result_object_detection_f32_config_t;

typedef struct {
    float threshold;
    uint8_t version;
    uint32_t object_detection_count;
    uint32_t output_features_count;
    float zero_point;
    float scale;
    voice_ei_object_detection_nms_config_t nms_config;
} voice_ei_fill_result_object_detection_i8_config_t;

typedef struct {
    float min_score_pixel;
    float min_score_box;
    float unclip_ratio;
    uint32_t object_detection_count;
    uint32_t output_features_count;
    voice_ei_object_detection_nms_config_t nms_config;
} voice_ei_fill_result_paddleocr_f32_config_t;

typedef struct {
    float min_score_pixel;
    float min_score_box;
    float unclip_ratio;
    uint32_t object_detection_count;
    uint32_t output_features_count;
    float zero_point;
    float scale;
    voice_ei_object_detection_nms_config_t nms_config;
} voice_ei_fill_result_paddleocr_i8_config_t;

typedef struct {
    float zero_point;
    float scale;
} voice_ei_fill_result_classification_i8_config_t;

typedef struct {
    float threshold;
    uint16_t out_width;
    uint16_t out_height;
    uint32_t object_detection_count;
} voice_ei_fill_result_fomo_f32_config_t;

typedef struct {
    float threshold;
    uint16_t out_width;
    uint16_t out_height;
    uint32_t object_detection_count;
    float zero_point;
    float scale;
} voice_ei_fill_result_fomo_i8_config_t;

typedef struct {
    float threshold;
    uint16_t grid_size_x;
    uint16_t grid_size_y;
} voice_ei_fill_result_visual_ad_f32_config_t;

// A struct which contains threshold descriptions (used in voice_ei_postprocessing_common.h)
typedef struct {
    std::string type;
    std::string name;
    float value;
    std::function<void(float)> set_value;
} voice_ei_threshold_desc_t;

template <typename>
struct voice_ei_dependent_false_v {
    enum { value = 0 };
};

#endif /* VOICE_EI_POSTPROCESSING_TYPES_H */
