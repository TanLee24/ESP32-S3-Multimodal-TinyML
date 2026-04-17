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

#ifndef _VOICE_EDGE_IMPULSE_RUN_CLASSIFIER_H_
#define _VOICE_EDGE_IMPULSE_RUN_CLASSIFIER_H_

#include "voice_ei_model_types.h"
#include "voice-model-parameters/model_metadata.h"

#include "voice_ei_run_dsp.h"
#include "voice_ei_classifier_types.h"
#include "voice_ei_signal_with_axes.h"
#include "postprocessing/voice_ei_postprocessing.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_data_normalization.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_print_results.h"

#include "voice-edge-impulse-sdk/porting/voice_ei_classifier_porting.h"
#include "voice-edge-impulse-sdk/porting/voice_ei_logging.h"
#include <memory>

#if VOICE_EI_CLASSIFIER_LOAD_ANOMALY_H
#include "inferencing_engines/anomaly.h"
#endif // VOICE_EI_CLASSIFIER_LOAD_ANOMALY_H

#if defined(VOICE_EI_CLASSIFIER_HAS_SAMPLER) && VOICE_EI_CLASSIFIER_HAS_SAMPLER == 1
#include "voice_ei_sampler.h"
#endif

#if (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE) && (VOICE_EI_CLASSIFIER_COMPILED != 1)
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tflite_micro.h"
#elif VOICE_EI_CLASSIFIER_COMPILED == 1
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tflite_eon.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE_FULL
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE_TIDL
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tflite_tidl.h"
#elif (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TENSORRT)
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tensorrt.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TENSAIFLOW
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tensaiflow.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/drpai.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_AKIDA
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/akida.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ONNX_TIDL
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/onnx_tidl.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_MEMRYX
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/memryx.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ETHOS_LINUX
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/ethos_linux.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ATON
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/aton.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_CEVA_NPN
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/ceva_npn.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_VLM_CONNECTOR
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/vlm_connector.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_NORDIC_AXON
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/nordic_axon.h"
#elif VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_NONE
// noop
#else
#error "Unknown inferencing engine"
#endif

// This file has an implicit dependency on voice_ei_run_dsp.h, so must come after that include!
#include "voice-model-parameters/model_variables.h"

#ifdef __cplusplus
namespace {
#endif // __cplusplus

/* Function prototypes ----------------------------------------------------- */
extern "C" VOICE_EI_IMPULSE_ERROR voice_run_inference(voice_ei_impulse_handle_t *handle, voice_ei_feature_t *fmatrix, voice_ei_impulse_result_t *result, bool debug);
extern "C" VOICE_EI_IMPULSE_ERROR voice_run_classifier_image_quantized(const voice_ei_impulse_t *impulse, signal_t *signal, voice_ei_impulse_result_t *result, bool debug);
static VOICE_EI_IMPULSE_ERROR can_voice_run_classifier_image_quantized(const voice_ei_impulse_t *impulse, voice_ei_learning_block_t block_ptr);
static void voice_ei_result_struct_timing_us_to_ms(voice_ei_impulse_result_t *result);

#if VOICE_EI_CLASSIFIER_LOAD_IMAGE_SCALING
VOICE_EI_IMPULSE_ERROR voice_ei_scale_fmatrix(voice_ei_learning_block_t *block, ei::matrix_t *fmatrix);
VOICE_EI_IMPULSE_ERROR voice_ei_unscale_fmatrix(voice_ei_learning_block_t *block, ei::matrix_t *fmatrix);
#endif // VOICE_EI_CLASSIFIER_LOAD_IMAGE_SCALING

/* Private variables ------------------------------------------------------- */

static uint64_t classifier_continuous_features_written = 0;

/* Private functions ------------------------------------------------------- */

/* These functions (up to Public functions section) are not exposed to end-user,
therefore changes are allowed. */

/**
 * @brief      Display the results of the inference
 *
 * @param      result  The result
 */
__attribute__((unused)) void display_results(voice_ei_impulse_handle_t *handle, voice_ei_impulse_result_t* result)
{
    voice_ei_print_results(handle, result);
    voice_display_postprocessing(handle, result);
}

/**
 * @brief      Do inferencing over the processed feature matrix
 *
 * @param      impulse  struct with information about model and DSP
 * @param      fmatrix  Processed matrix
 * @param      result   Output classifier results
 * @param[in]  debug    Debug output enable
 *
 * @return     The ei impulse error.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_run_inference(
    voice_ei_impulse_handle_t *handle,
    voice_ei_feature_t *fmatrix,
    voice_ei_impulse_result_t *result,
    bool debug = false)
{
    auto& impulse = handle->impulse;
    for (size_t ix = 0; ix < impulse->learning_blocks_size; ix++) {

        voice_ei_learning_block_t block = impulse->learning_blocks[ix];

#if VOICE_EI_CLASSIFIER_LOAD_IMAGE_SCALING
        auto start_scale_matrix_us = voice_ei_read_timer_us();

        // we do not plan to have multiple dsp blocks with image
        // so just apply scaling to the first one
        VOICE_EI_IMPULSE_ERROR scale_res = voice_ei_scale_fmatrix(&block, fmatrix[0].matrix);
        if (scale_res != VOICE_EI_IMPULSE_OK) {
            return scale_res;
        }

        auto end_scale_matrix_us = voice_ei_read_timer_us();
#endif

        VOICE_EI_IMPULSE_ERROR res = block.infer_fn(impulse, fmatrix, ix, (uint32_t*)block.input_block_ids, block.input_block_ids_size, result, block.config, debug);
        if (res != VOICE_EI_IMPULSE_OK) {
            return res;
        }

#if VOICE_EI_CLASSIFIER_LOAD_IMAGE_SCALING
        auto start_unscale_matrix_us = voice_ei_read_timer_us();

        // undo scaling, only if we have multiple learn blocks... otherwise just leave scaled
        if (impulse->learning_blocks_size > 1) {
            scale_res = voice_ei_unscale_fmatrix(&block, fmatrix[0].matrix);
            if (scale_res != VOICE_EI_IMPULSE_OK) {
                return scale_res;
            }
        }

        auto end_unscale_matrix_us = voice_ei_read_timer_us();

        // count scaling in the DSP timing
        result->timing.dsp_us += (end_unscale_matrix_us - start_unscale_matrix_us) +
                                 (end_scale_matrix_us - start_scale_matrix_us);
#endif
    }

    if (voice_ei_run_impulse_check_canceled() == VOICE_EI_IMPULSE_CANCELED) {
        return VOICE_EI_IMPULSE_CANCELED;
    }

    return VOICE_EI_IMPULSE_OK;
}

/**
 * @brief      Process a complete impulse
 *
 * @param      impulse  struct with information about model and DSP
 * @param      signal   Sample data
 * @param      result   Output classifier results
 * @param      handle   Handle from open_impulse. nullptr for backward compatibility
 * @param[in]  debug    Debug output enable
 *
 * @return     The ei impulse error.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_process_impulse(voice_ei_impulse_handle_t *handle,
                                            signal_t *signal,
                                            voice_ei_impulse_result_t *result,
                                            bool debug = false)
{
    if ((handle == nullptr) || (handle->impulse  == nullptr) || (result  == nullptr) || (signal  == nullptr)) {
        return VOICE_EI_IMPULSE_INFERENCE_ERROR;
    }

    memset(result, 0, sizeof(voice_ei_impulse_result_t));

#if VOICE_EI_IMPULSE_RESULT_CLASSIFICATION_IS_STATICALLY_ALLOCATED == 0
    static std::vector<voice_ei_impulse_result_classification_t> classification_results;
    classification_results.clear(); // todo, should not clear and re-gen this every time...

    if (handle->impulse->results_type == VOICE_EI_CLASSIFIER_TYPE_CLASSIFICATION ||
        handle->impulse->results_type == VOICE_EI_CLASSIFIER_TYPE_REGRESSION) {
    #ifdef VOICE_EI_DSP_RESULT_OVERRIDE
        for (size_t ix = 0; ix < VOICE_EI_DSP_RESULT_OVERRIDE; ix++) {
            voice_ei_impulse_result_classification_t classification = {
                .label = "",
                .value = 0.0f
            };
            classification_results.push_back(classification);
        }
    #else
        for (size_t ix = 0; ix < handle->impulse->label_count; ix++) {
            voice_ei_impulse_result_classification_t classification = {
                .label = handle->impulse->categories[ix],
                .value = 0.0f
            };
            classification_results.push_back(classification);
        }
    #endif // VOICE_EI_DSP_RESULT_OVERRIDE
    }

    result->classification = classification_results.data();
#endif // VOICE_EI_IMPULSE_RESULT_CLASSIFICATION_IS_STATICALLY_ALLOCATED == 0

    uint8_t num_results = handle->impulse->output_tensors_size;

    std::unique_ptr<voice_ei_feature_t[]> raw_results_ptr(new voice_ei_feature_t[num_results]);

    result->_raw_outputs = raw_results_ptr.get();
    memset(result->_raw_outputs, 0, sizeof(voice_ei_feature_t) * num_results);

    VOICE_EI_IMPULSE_ERROR res = VOICE_EI_IMPULSE_OK;
    (void)res; // Get around -Werror=unused-variable if neither of the calls below are compiled in (e.g. unit-tests/hr)

#if (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_VLM_CONNECTOR)
    // Shortcut for vlm models
    res = run_vlm_inference(handle, signal, 0, result, handle->impulse->learning_blocks[0].config, false);
    if (res != VOICE_EI_IMPULSE_OK) {
        return res;
    }
    res = voice_run_postprocessing(handle, result);
    return res;
#endif // VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_VLM_CONNECTOR
#if (VOICE_EI_CLASSIFIER_QUANTIZATION_ENABLED == 1 && (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TENSAIFLOW || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ONNX_TIDL) || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ATON)
    // Shortcut for quantized image models
    voice_ei_learning_block_t block = handle->impulse->learning_blocks[0];
    if (can_voice_run_classifier_image_quantized(handle->impulse, block) == VOICE_EI_IMPULSE_OK) {
        res = voice_run_classifier_image_quantized(handle->impulse, signal, result, debug);
        if (res != VOICE_EI_IMPULSE_OK) {
            return res;
        }
        res = voice_run_postprocessing(handle, result);
        voice_ei_result_struct_timing_us_to_ms(result);
        return res;
    }
#endif // VOICE_EI_CLASSIFIER_QUANTIZATION_ENABLED == 1 && (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TENSAIFLOW || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ONNX_TIDL) || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ATON
    uint32_t block_num = handle->impulse->dsp_blocks_size;

    // smart pointer to features array
    std::unique_ptr<voice_ei_feature_t[]> features_ptr(new voice_ei_feature_t[block_num]);
    voice_ei_feature_t* features = features_ptr.get();

    if (features == nullptr) {
        voice_ei_printf("ERR: Out of memory, can't allocate features\n");
        return VOICE_EI_IMPULSE_ALLOC_FAILED;
    }

    memset(features, 0, sizeof(voice_ei_feature_t) * block_num);

    // have it outside of the loop to avoid going out of scope
    std::unique_ptr<std::unique_ptr<ei::matrix_t>[]> matrix_ptrs_ptr(new std::unique_ptr<ei::matrix_t>[block_num]);
    std::unique_ptr<ei::matrix_t> *matrix_ptrs = matrix_ptrs_ptr.get();

    if (matrix_ptrs == nullptr) {
        delete[] matrix_ptrs;
        voice_ei_printf("ERR: Out of memory, can't allocate matrix_ptrs\n");
        return VOICE_EI_IMPULSE_ALLOC_FAILED;
    }

    uint64_t dsp_start_us = voice_ei_read_timer_us();

    size_t out_features_index = 0;

    for (size_t ix = 0; ix < handle->impulse->dsp_blocks_size; ix++) {
        voice_ei_model_dsp_t block = handle->impulse->dsp_blocks[ix];

        matrix_ptrs[ix] = std::unique_ptr<ei::matrix_t>(new ei::matrix_t(1, block.n_output_features));
        if (matrix_ptrs[ix] == nullptr) {
            voice_ei_printf("ERR: Out of memory, can't allocate matrix_ptrs[%lu]\n", (unsigned long)ix);
            return VOICE_EI_IMPULSE_ALLOC_FAILED;
        }

        if (matrix_ptrs[ix]->buffer == nullptr) {
            voice_ei_printf("ERR: Out of memory, can't allocate matrix_ptrs[%lu]\n", (unsigned long)ix);
            delete[] matrix_ptrs;
            return VOICE_EI_IMPULSE_ALLOC_FAILED;
        }

        features[ix].matrix = matrix_ptrs[ix].get();
        features[ix].blockId = block.blockId;

        if (out_features_index + block.n_output_features > handle->impulse->nn_input_frame_size) {
            voice_ei_printf("ERR: Would write outside feature buffer\n");
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }

#if EIDSP_SIGNAL_C_FN_POINTER
        if (block.axes_size != handle->impulse->raw_samples_per_frame) {
            voice_ei_printf("ERR: EIDSP_SIGNAL_C_FN_POINTER can only be used when all axes are selected for DSP blocks\n");
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
        auto internal_signal = signal;
#else
        SignalWithAxes swa(signal, block.axes, block.axes_size, handle->impulse);
        auto internal_signal = swa.get_signal();
#endif

        int ret;
        if (block.factory) { // ie, if we're using state
            // Msg user
            static bool has_printed = false;
            if (!has_printed) {
                VOICE_EI_LOGI("Impulse maintains state. Call voice_run_classifier_init() to reset state (e.g. if data stream is interrupted.)\n");
                has_printed = true;
            }

            // getter has a lazy init, so we can just call it
            auto dsp_handle = handle->state.get_dsp_handle(ix);
            if(dsp_handle) {
                ret = dsp_handle->extract(
                    internal_signal,
                    features[ix].matrix,
                    block.config,
                    handle->impulse->frequency,
                    result);
            }
            else {
                return VOICE_EI_IMPULSE_OUT_OF_MEMORY;
            }
        } else {
            ret = block.extract_fn(internal_signal, features[ix].matrix, block.config, handle->impulse->frequency);
        }

        if (ret != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }

        if (voice_ei_run_impulse_check_canceled() == VOICE_EI_IMPULSE_CANCELED) {
            return VOICE_EI_IMPULSE_CANCELED;
        }

        out_features_index += block.n_output_features;
    }

#if VOICE_EI_CLASSIFIER_HAS_DATA_NORMALIZATION
    VOICE_EI_IMPULSE_ERROR dn_error = run_data_normalization(handle, features);
    if (dn_error != VOICE_EI_IMPULSE_OK) {
        voice_ei_printf("ERR: Failed to run Data Normalization process (%d)\n", dn_error);
        return dn_error;
    }
#endif

    result->timing.dsp_us = voice_ei_read_timer_us() - dsp_start_us;

    if (debug) {
        voice_ei_printf("Features (%d ms.): ", result->timing.dsp);
        for (size_t ix = 0; ix < block_num; ix++) {
            if (features[ix].matrix == nullptr) {
                continue;
            }
            for (size_t jx = 0; jx < features[ix].matrix->cols; jx++) {
                voice_ei_printf_float(features[ix].matrix->buffer[jx]);
                voice_ei_printf(" ");
            }
            voice_ei_printf("\n");
        }
    }

    if (debug) {
        voice_ei_printf("Running impulse...\n");
    }

#if VOICE_EI_CLASSIFIER_DSP_ONLY
    voice_ei_result_struct_timing_us_to_ms(result);

    return VOICE_EI_IMPULSE_OK;
#else
    res = voice_run_inference(handle, features, result, debug);
    if (res != VOICE_EI_IMPULSE_OK) {
        return res;
    }

    res = voice_run_postprocessing(handle, result);
    if (res != VOICE_EI_IMPULSE_OK) {
        return res;
    }

    voice_ei_result_struct_timing_us_to_ms(result);

    return VOICE_EI_IMPULSE_OK;
#endif
}

/**
 * @brief      Opens an impulse
 *
 * @param      impulse  struct with information about model and DSP
 *
 * @return     A pointer to the impulse handle, or nullptr if memory allocation failed.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_init_impulse(voice_ei_impulse_handle_t *handle) {
    if (!handle) {
        return VOICE_EI_IMPULSE_OUT_OF_MEMORY;
    }
    handle->state.reset();
    return VOICE_EI_IMPULSE_OK;
}

/**
 * @brief      Process a complete impulse for continuous inference
 *
 * @param      handle               struct with information about model and DSP
 * @param      signal               Sample data
 * @param      result               Output classifier results
 * @param[in]  debug                Debug output enable
 *
 * @return     The ei impulse error.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_process_impulse_continuous(voice_ei_impulse_handle_t *handle,
                                                       signal_t *signal,
                                                       voice_ei_impulse_result_t *result,
                                                       bool debug = false)
{
    if ((handle == nullptr) || (handle->impulse  == nullptr) || (result  == nullptr) || (signal  == nullptr)) {
        return VOICE_EI_IMPULSE_INFERENCE_ERROR;
    }

    memset(result, 0, sizeof(voice_ei_impulse_result_t));

#if VOICE_EI_IMPULSE_RESULT_CLASSIFICATION_IS_STATICALLY_ALLOCATED == 0
    static std::vector<voice_ei_impulse_result_classification_t> classification_results;
    classification_results.clear(); // todo, should not clear and re-gen this every time...

    if (handle->impulse->results_type == VOICE_EI_CLASSIFIER_TYPE_CLASSIFICATION ||
        handle->impulse->results_type == VOICE_EI_CLASSIFIER_TYPE_REGRESSION) {
    #ifdef VOICE_EI_DSP_RESULT_OVERRIDE
        for (size_t ix = 0; ix < VOICE_EI_DSP_RESULT_OVERRIDE; ix++) {
            voice_ei_impulse_result_classification_t classification = {
                .label = "",
                .value = 0.0f
            };
            classification_results.push_back(classification);
        }
    #else
        for (size_t ix = 0; ix < handle->impulse->label_count; ix++) {
            voice_ei_impulse_result_classification_t classification = {
                .label = handle->impulse->categories[ix],
                .value = 0.0f
            };
            classification_results.push_back(classification);
        }
    #endif
    }

    result->classification = classification_results.data();

#else // VOICE_EI_IMPULSE_RESULT_CLASSIFICATION_IS_STATICALLY_ALLOCATED == 1

    for (int i = 0; i < handle->impulse->label_count; i++) {
        // set label correctly in the result struct if we have no results (otherwise is nullptr)
        result->classification[i].label = handle->impulse->categories[(uint32_t)i];
    }

#endif // VOICE_EI_IMPULSE_RESULT_CLASSIFICATION_IS_STATICALLY_ALLOCATED == 0

    // smart pointer to results array
    std::unique_ptr<voice_ei_feature_t[]> raw_results_ptr(new voice_ei_feature_t[handle->impulse->learning_blocks_size]);
    result->_raw_outputs = raw_results_ptr.get();
    memset(result->_raw_outputs, 0, sizeof(voice_ei_feature_t) * handle->impulse->learning_blocks_size);

    auto impulse = handle->impulse;
    static ei::matrix_t static_features_matrix(1, impulse->nn_input_frame_size);
    if (!static_features_matrix.buffer) {
        return VOICE_EI_IMPULSE_ALLOC_FAILED;
    }

    VOICE_EI_IMPULSE_ERROR voice_ei_impulse_error = VOICE_EI_IMPULSE_OK;

    uint64_t dsp_start_us = voice_ei_read_timer_us();

    size_t out_features_index = 0;

    for (size_t ix = 0; ix < impulse->dsp_blocks_size; ix++) {
        voice_ei_model_dsp_t block = impulse->dsp_blocks[ix];

        if (out_features_index + block.n_output_features > impulse->nn_input_frame_size) {
            voice_ei_printf("ERR: Would write outside feature buffer\n");
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }

        ei::matrix_t fm(1, block.n_output_features,
                        static_features_matrix.buffer + out_features_index);

        int (*extract_fn_slice)(ei::signal_t *signal, ei::matrix_t *output_matrix, void *config, const float frequency, matrix_size_t *out_matrix_size);

        /* Switch to the slice version of the mfcc feature extract function */
        if (block.extract_fn == extract_mfcc_features) {
            extract_fn_slice = &extract_mfcc_per_slice_features;
        }
        else if (block.extract_fn == extract_spectrogram_features) {
            extract_fn_slice = &extract_spectrogram_per_slice_features;
        }
        else if (block.extract_fn == extract_mfe_features) {
            extract_fn_slice = &extract_mfe_per_slice_features;
        }
        else {
            voice_ei_printf("ERR: Unknown extract function, only MFCC, MFE and spectrogram supported\n");
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }

        matrix_size_t features_written;

#if EIDSP_SIGNAL_C_FN_POINTER
        if (block.axes_size != impulse->raw_samples_per_frame) {
            voice_ei_printf("ERR: EIDSP_SIGNAL_C_FN_POINTER can only be used when all axes are selected for DSP blocks\n");
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
        int ret = extract_fn_slice(signal, &fm, block.config, impulse->frequency, &features_written);
#else
        SignalWithAxes swa(signal, block.axes, block.axes_size, impulse);
        int ret = extract_fn_slice(swa.get_signal(), &fm, block.config, impulse->frequency, &features_written);
#endif

        if (ret != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }

        if (voice_ei_run_impulse_check_canceled() == VOICE_EI_IMPULSE_CANCELED) {
            return VOICE_EI_IMPULSE_CANCELED;
        }

        classifier_continuous_features_written += (features_written.rows * features_written.cols);

        out_features_index += block.n_output_features;
    }

    result->timing.dsp_us = voice_ei_read_timer_us() - dsp_start_us;

    if (classifier_continuous_features_written >= impulse->nn_input_frame_size) {
        dsp_start_us = voice_ei_read_timer_us();

        uint32_t block_num = impulse->dsp_blocks_size + impulse->learning_blocks_size;

        // smart pointer to features array
        std::unique_ptr<voice_ei_feature_t[]> features_ptr(new voice_ei_feature_t[block_num]);
        voice_ei_feature_t* features = features_ptr.get();
        if (features == nullptr) {
            voice_ei_printf("ERR: Out of memory, can't allocate features\n");
            return VOICE_EI_IMPULSE_ALLOC_FAILED;
        }
        memset(features, 0, sizeof(voice_ei_feature_t) * block_num);

        // have it outside of the loop to avoid going out of scope
        std::unique_ptr<ei::matrix_t> *matrix_ptrs = new std::unique_ptr<ei::matrix_t>[block_num];
        if (matrix_ptrs == nullptr) {
            voice_ei_printf("ERR: Out of memory, can't allocate matrix_ptrs\n");
            return VOICE_EI_IMPULSE_ALLOC_FAILED;
        }

        out_features_index = 0;
        // iterate over every dsp block and run normalization
        for (size_t ix = 0; ix < impulse->dsp_blocks_size; ix++) {
            voice_ei_model_dsp_t block = impulse->dsp_blocks[ix];
            matrix_ptrs[ix] = std::unique_ptr<ei::matrix_t>(new ei::matrix_t(1, block.n_output_features));

            if (matrix_ptrs[ix] == nullptr) {
                voice_ei_printf("ERR: Out of memory, can't allocate matrix_ptrs[%lu]\n", (unsigned long)ix);
                return VOICE_EI_IMPULSE_ALLOC_FAILED;
            }

            if (matrix_ptrs[ix]->buffer == nullptr) {
                voice_ei_printf("ERR: Out of memory, can't allocate matrix_ptrs[%lu]\n", (unsigned long)ix);
                delete[] matrix_ptrs;
                return VOICE_EI_IMPULSE_ALLOC_FAILED;
            }

            features[ix].matrix = matrix_ptrs[ix].get();
            features[ix].blockId = block.blockId;

            /* Create a copy of the matrix for normalization */
            for (size_t m_ix = 0; m_ix < block.n_output_features; m_ix++) {
                features[ix].matrix->buffer[m_ix] = static_features_matrix.buffer[out_features_index + m_ix];
            }

            if (block.extract_fn == extract_mfcc_features) {
                calc_cepstral_mean_and_var_normalization_mfcc(features[ix].matrix, block.config);
            }
            else if (block.extract_fn == extract_spectrogram_features) {
                calc_cepstral_mean_and_var_normalization_spectrogram(features[ix].matrix, block.config);
            }
            else if (block.extract_fn == extract_mfe_features) {
                calc_cepstral_mean_and_var_normalization_mfe(features[ix].matrix, block.config);
            }
            out_features_index += block.n_output_features;
        }

        result->timing.dsp_us += voice_ei_read_timer_us() - dsp_start_us;

        if (debug) {
            voice_ei_printf("Feature Matrix: \n");
            for (size_t ix = 0; ix < features->matrix->cols; ix++) {
                voice_ei_printf_float(features->matrix->buffer[ix]);
                voice_ei_printf(" ");
            }
            voice_ei_printf("\n");
            voice_ei_printf("Running impulse...\n");
        }

        voice_ei_impulse_error = voice_run_inference(handle, features, result, debug);
        if (voice_ei_impulse_error != VOICE_EI_IMPULSE_OK) {
            return voice_ei_impulse_error;
        }
        delete[] matrix_ptrs;
        voice_ei_impulse_error = voice_run_postprocessing(handle, result);
        if (voice_ei_impulse_error != VOICE_EI_IMPULSE_OK) {
            return voice_ei_impulse_error;
        }
    }

    voice_ei_result_struct_timing_us_to_ms(result);

    return voice_ei_impulse_error;
}

/**
 * Check if the current impulse could be used by 'voice_run_classifier_image_quantized'
 */
__attribute__((unused)) static VOICE_EI_IMPULSE_ERROR can_voice_run_classifier_image_quantized(const voice_ei_impulse_t *impulse, voice_ei_learning_block_t block_ptr) {

    if (impulse->inferencing_engine != VOICE_EI_CLASSIFIER_TFLITE
        && impulse->inferencing_engine != VOICE_EI_CLASSIFIER_TENSAIFLOW
        && impulse->inferencing_engine != VOICE_EI_CLASSIFIER_DRPAI
        && impulse->inferencing_engine != VOICE_EI_CLASSIFIER_ONNX_TIDL
        && impulse->inferencing_engine != VOICE_EI_CLASSIFIER_ATON) // check later
    {
        return VOICE_EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE;
    }

    // visual anomaly also needs to go through the normal path
    if (impulse->has_anomaly){
        return VOICE_EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
    }

        // Check if we have tflite graph
    if (block_ptr.infer_fn != run_nn_inference) {
        return VOICE_EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
    }

    // Check if we have a quantized NN Input layer (input is always quantized for DRP-AI)
    voice_ei_learning_block_config_tflite_graph_t *block_config = (voice_ei_learning_block_config_tflite_graph_t*)block_ptr.config;
    if (block_config->quantized != 1) {
        return VOICE_EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
    }

    // And if we have one DSP block which operates on images...
    if (impulse->dsp_blocks_size != 1 || impulse->dsp_blocks[0].extract_fn != extract_image_features) {
        return VOICE_EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
    }

    return VOICE_EI_IMPULSE_OK;
}

#if VOICE_EI_CLASSIFIER_QUANTIZATION_ENABLED == 1 && (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TENSAIFLOW || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ONNX_TIDL || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_ATON)

/**
 * Special function to run the classifier on images, only works on TFLite models (either interpreter, EON, tensaiflow, drpai, tidl, memryx)
 * that allocates a lot less memory by quantizing in place. This only works if 'can_voice_run_classifier_image_quantized'
 * returns VOICE_EI_IMPULSE_OK.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_run_classifier_image_quantized(
    const voice_ei_impulse_t *impulse,
    signal_t *signal,
    voice_ei_impulse_result_t *result,
    bool debug = false)
{
    return run_nn_inference_image_quantized(impulse, signal, 0, result, impulse->learning_blocks[0].config, debug);
}

#endif // #if VOICE_EI_CLASSIFIER_QUANTIZATION_ENABLED == 1 && (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TENSAIFLOW || VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI)

#if VOICE_EI_CLASSIFIER_LOAD_IMAGE_SCALING
static const float torch_mean[] = { 0.485, 0.456, 0.406 };
static const float torch_std[] = { 0.229, 0.224, 0.225 };
// This is ordered BGR
static const float tao_mean[] = { 103.939, 116.779, 123.68 };

VOICE_EI_IMPULSE_ERROR voice_ei_scale_fmatrix(voice_ei_learning_block_t *block, ei::matrix_t *fmatrix) {
    if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_TORCH) {
        // @todo; could we write some faster vector math here?
        for (size_t ix = 0; ix < fmatrix->rows * fmatrix->cols; ix += 3) {
            fmatrix->buffer[ix + 0] = (fmatrix->buffer[ix + 0] - torch_mean[0]) / torch_std[0];
            fmatrix->buffer[ix + 1] = (fmatrix->buffer[ix + 1] - torch_mean[1]) / torch_std[1];
            fmatrix->buffer[ix + 2] = (fmatrix->buffer[ix + 2] - torch_mean[2]) / torch_std[2];
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_0_255) {
        int scale_res = numpy::scale(fmatrix, 255.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_MIN128_127) {
        int scale_res = numpy::scale_and_add(fmatrix, 255.0f, -128.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_MIN1_1) {
        int scale_res = numpy::scale_and_add(fmatrix, 2.0f, -1.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_BGR_SUBTRACT_IMAGENET_MEAN) {
        int scale_res = numpy::scale(fmatrix, 255.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
        // Transpose RGB to BGR and subtract mean
        for (size_t ix = 0; ix < fmatrix->rows * fmatrix->cols; ix += 3) {
            float r = fmatrix->buffer[ix + 0];
            fmatrix->buffer[ix + 0] = fmatrix->buffer[ix + 2] - tao_mean[0];
            fmatrix->buffer[ix + 1] -= tao_mean[1];
            fmatrix->buffer[ix + 2] = r - tao_mean[2];
        }
    }

    return VOICE_EI_IMPULSE_OK;
}

VOICE_EI_IMPULSE_ERROR voice_ei_unscale_fmatrix(voice_ei_learning_block_t *block, ei::matrix_t *fmatrix) {
    if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_TORCH) {
        // @todo; could we write some faster vector math here?
        for (size_t ix = 0; ix < fmatrix->rows * fmatrix->cols; ix += 3) {
            fmatrix->buffer[ix + 0] = (fmatrix->buffer[ix + 0] * torch_std[0]) + torch_mean[0];
            fmatrix->buffer[ix + 1] = (fmatrix->buffer[ix + 1] * torch_std[1]) + torch_mean[1];
            fmatrix->buffer[ix + 2] = (fmatrix->buffer[ix + 2] * torch_std[2]) + torch_mean[2];
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_MIN128_127) {
        int scale_res = numpy::scale_and_add(fmatrix, 1.0f / 255.0f, 128.0f / 255.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_MIN1_1) {
        int scale_res = numpy::scale_and_add(fmatrix, 1.0f / 2.0f, 1.0f / 2.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_0_255) {
        int scale_res = numpy::scale(fmatrix, 1 / 255.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    else if (block->image_scaling == VOICE_EI_CLASSIFIER_IMAGE_SCALING_BGR_SUBTRACT_IMAGENET_MEAN) {
        // Transpose BGR to RGB and add mean
        for (size_t ix = 0; ix < fmatrix->rows * fmatrix->cols; ix += 3) {
            float b = fmatrix->buffer[ix + 0];
            fmatrix->buffer[ix + 0] = fmatrix->buffer[ix + 2] + tao_mean[2];
            fmatrix->buffer[ix + 1] += tao_mean[1];
            fmatrix->buffer[ix + 2] = b + tao_mean[0];
        }
        int scale_res = numpy::scale(fmatrix, 1 / 255.0f);
        if (scale_res != EIDSP_OK) {
            voice_ei_printf("ERR: Failed to scale matrix (%d)\n", scale_res);
            return VOICE_EI_IMPULSE_DSP_ERROR;
        }
    }
    return VOICE_EI_IMPULSE_OK;
}
#endif

/**
 * Internally we store data in the timing.*_us fields -> sync them to the non-us fields
 * as users might use those instead.
 */
static void voice_ei_result_struct_timing_us_to_ms(voice_ei_impulse_result_t *result) {
    // This does the same as:
    //   result->timing.dsp = (int)round((float)result->timing.dsp_us / 1000.0f);
    // but this requires floating point math (e.g. loads in _arm_addsubsf3.o -> ~600 extra bytes flash)

    result->timing.dsp = (int)((result->timing.dsp_us + 500) / 1000);
    result->timing.classification = (int)((result->timing.classification_us + 500) / 1000);
    result->timing.anomaly = (int)((result->timing.anomaly_us + 500) / 1000);
    result->timing.postprocessing = (int)((result->timing.postprocessing_us + 500) / 1000);
}

/* Public functions ------------------------------------------------------- */

/* Tread carefully: public functions are not to be changed
to preserve backwards compatibility. Anything in this public section
will be documented by Doxygen. */

/**
 * @defgroup voice_ei_functions Functions
 *
 * Public-facing functions for running inference using the Edge Impulse C++ library.
 *
 * **Source**: [classifier/voice_ei_voice_run_classifier.h](https://github.com/edgeimpulse/inferencing-sdk-cpp/blob/master/classifier/voice_ei_voice_run_classifier.h)
 *
 * @addtogroup voice_ei_functions
 * @{
 */

/**
 * @brief Initialize static variables for running preprocessing and inference
 *  continuously.
 *
 * Initializes and clears any internal static variables needed by `voice_run_classifier_continuous()`.
 * This includes the moving average filter (MAF). This function should be called prior to
 * calling `voice_run_classifier_continuous()`.
 *
 * **Blocking**: yes
 *
 * **Example**: [nano_ble33_sense_microphone_continuous.ino](https://github.com/edgeimpulse/example-lacuna-ls200/blob/main/nano_ble33_sense_microphone_continous/nano_ble33_sense_microphone_continuous.ino)
 */
extern "C" void voice_run_classifier_init(void)
{

    classifier_continuous_features_written = 0;
    voice_ei_dsp_clear_continuous_audio_state();
    voice_init_impulse(&voice_ei_default_impulse);
    voice_init_postprocessing(&voice_ei_default_impulse);
#if VOICE_EI_CLASSIFIER_HAS_DATA_NORMALIZATION
    init_data_normalization(&voice_ei_default_impulse);
#endif
}

/**
 * @brief Initialize static variables for running preprocessing and inference
 *  continuously.
 *
 * Initializes and clears any internal static variables needed by `voice_run_classifier_continuous()`.
 * This includes the moving average filter (MAF). This function should be called prior to
 * calling `voice_run_classifier_continuous()`.
 *
 * **Blocking**: yes
 *
 * **Example**: [nano_ble33_sense_microphone_continuous.ino](https://github.com/edgeimpulse/example-lacuna-ls200/blob/main/nano_ble33_sense_microphone_continous/nano_ble33_sense_microphone_continuous.ino)
 *
 * @param[in]   handle struct with information about model and DSP
 */
__attribute__((unused)) void voice_run_classifier_init(voice_ei_impulse_handle_t *handle)
{
    classifier_continuous_features_written = 0;
    voice_ei_dsp_clear_continuous_audio_state();
    voice_init_impulse(handle);
    voice_init_postprocessing(handle);
#if VOICE_EI_CLASSIFIER_HAS_DATA_NORMALIZATION
    init_data_normalization(handle);
#endif
}

/**
 * @brief Deletes static variables when running preprocessing and inference continuously.
 *
 * Deletes internal static variables used by `voice_run_classifier_continuous()`, which
 * includes the moving average filter (MAF). This function should be called when you
 * are done running continuous classification.
 *
 * **Blocking**: yes
 *
 * **Example**: [voice_ei_run_audio_impulse.cpp](https://github.com/edgeimpulse/firmware-nordic-thingy53/blob/main/src/inference/voice_ei_run_audio_impulse.cpp)
 */
extern "C" void voice_run_classifier_deinit(void)
{
    devoice_init_postprocessing(&voice_ei_default_impulse);
}

__attribute__((unused)) void voice_run_classifier_deinit(voice_ei_impulse_handle_t *handle)
{
    devoice_init_postprocessing(handle);
#if VOICE_EI_CLASSIFIER_HAS_DATA_NORMALIZATION
    deinit_data_normalization(handle);
#endif
}

/**
 * @brief Run preprocessing (DSP) on new slice of raw features. Add output features
 *  to rolling matrix and run inference on full sample.
 *
 * Accepts a new slice of features give by the callback defined in the `signal` parameter.
 * It performs preprocessing (DSP) on this new slice of features and appends the output to
 * a sliding window of pre-processed features (stored in a static features matrix). The matrix
 * stores the new slice and as many old slices as necessary to make up one full sample for
 * performing inference.
 *
 * `voice_run_classifier_init()` must be called before making any calls to
 * `voice_run_classifier_continuous().`
 *
 * For example, if you are doing keyword spotting on 1-second slices of audio and you want to
 * perform inference 4 times per second (given by `VOICE_EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW`), you
 * would collect 0.25 seconds of audio and call voice_run_classifier_continuous(). The function would
 * compute the Mel-Frequency Cepstral Coefficients (MFCCs) for that 0.25 second slice of audio,
 * drop the oldest 0.25 seconds' worth of MFCCs from its internal matrix, and append the newest
 * slice of MFCCs. This process allows the library to keep track of the pre-processed features
 * (e.g. MFCCs) in the window instead of the entire set of raw features (e.g. raw audio data),
 * which can potentially save a lot of space in RAM. After updating the static matrix,
 * inference is performed using the whole matrix, which acts as a sliding window of
 * pre-processed features.
 *
 * Additionally, a moving average filter (MAF) can be enabled for `voice_run_classifier_continuous()`,
 * which averages (arithmetic mean) the last *n* inference results for each class. *n* is
 * `VOICE_EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW / 2`. In our example above, if we enabled the MAF, the
 * values in `result` would contain predictions averaged from the previous 2 inferences.
 *
 * To learn more about `voice_run_classifier_continuous()`, see
 * [this guide](https://docs.edgeimpulse.com/docs/tutorials/advanced-inferencing/continuous-audio-sampling)
 * on continuous audio sampling. While the guide is written for audio signals, the concepts of continuous sampling and inference can be extrapolated to any time-series data.
 *
 * **Blocking**: yes
 *
 * **Example**: [nano_ble33_sense_microphone_continuous.ino](https://github.com/edgeimpulse/example-lacuna-ls200/blob/main/nano_ble33_sense_microphone_continous/nano_ble33_sense_microphone_continuous.ino)
 *
 * @param[in] signal  Pointer to a signal_t struct that contains the number of elements in the
 *  slice of raw features (e.g. `VOICE_EI_CLASSIFIER_SLICE_SIZE`) and a pointer to a callback that reads
 *  in the slice of raw features.
 * @param[out] result Pointer to an `voice_ei_impulse_result_t` struct that contains the various output
 *  results from inference after voice_run_classifier() returns.
 * @param[in]  debug Print internal preprocessing and inference debugging information via
 *  `voice_ei_printf()`.
 * @param[in]  enable_maf_unused Enable the moving average filter (MAF) for the classifier - deprecated, replaced with Performance Calibration
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if inference
 *  completed successfully.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_run_classifier_continuous(
    signal_t *signal,
    voice_ei_impulse_result_t *result,
    bool debug = false,
    bool enable_maf_unused = true)
{
    auto& impulse = voice_ei_default_impulse;
    return voice_process_impulse_continuous(&impulse, signal, result, debug);
}

/**
 * @brief Run preprocessing (DSP) on new slice of raw features. Add output features
 *  to rolling matrix and run inference on full sample.
 *
 * Accepts a new slice of features give by the callback defined in the `signal` parameter.
 * It performs preprocessing (DSP) on this new slice of features and appends the output to
 * a sliding window of pre-processed features (stored in a static features matrix). The matrix
 * stores the new slice and as many old slices as necessary to make up one full sample for
 * performing inference.
 *
 * `voice_run_classifier_init()` must be called before making any calls to
 * `voice_run_classifier_continuous().`
 *
 * For example, if you are doing keyword spotting on 1-second slices of audio and you want to
 * perform inference 4 times per second (given by `VOICE_EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW`), you
 * would collect 0.25 seconds of audio and call voice_run_classifier_continuous(). The function would
 * compute the Mel-Frequency Cepstral Coefficients (MFCCs) for that 0.25 second slice of audio,
 * drop the oldest 0.25 seconds' worth of MFCCs from its internal matrix, and append the newest
 * slice of MFCCs. This process allows the library to keep track of the pre-processed features
 * (e.g. MFCCs) in the window instead of the entire set of raw features (e.g. raw audio data),
 * which can potentially save a lot of space in RAM. After updating the static matrix,
 * inference is performed using the whole matrix, which acts as a sliding window of
 * pre-processed features.
 *
 * Additionally, a moving average filter (MAF) can be enabled for `voice_run_classifier_continuous()`,
 * which averages (arithmetic mean) the last *n* inference results for each class. *n* is
 * `VOICE_EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW / 2`. In our example above, if we enabled the MAF, the
 * values in `result` would contain predictions averaged from the previous 2 inferences.
 *
 * To learn more about `voice_run_classifier_continuous()`, see
 * [this guide](https://docs.edgeimpulse.com/docs/tutorials/advanced-inferencing/continuous-audio-sampling)
 * on continuous audio sampling. While the guide is written for audio signals, the concepts of continuous sampling and inference can be extrapolated to any time-series data.
 *
 * **Blocking**: yes
 *
 * **Example**: [nano_ble33_sense_microphone_continuous.ino](https://github.com/edgeimpulse/example-lacuna-ls200/blob/main/nano_ble33_sense_microphone_continous/nano_ble33_sense_microphone_continuous.ino)
 *
 * @param[in] impulse `voice_ei_impulse_handle_t` struct with information about preprocessing and model.
 * @param[in] signal  Pointer to a signal_t struct that contains the number of elements in the
 *  slice of raw features (e.g. `VOICE_EI_CLASSIFIER_SLICE_SIZE`) and a pointer to a callback that reads
 *  in the slice of raw features.
 * @param[out] result Pointer to an `voice_ei_impulse_result_t` struct that contains the various output
 *  results from inference after voice_run_classifier() returns.
 * @param[in] debug Print internal preprocessing and inference debugging information via
 *  `voice_ei_printf()`.
 * @param[in] enable_maf_unused Enable the moving average filter (MAF) for the classifier - deprecated, replaced with Performance Calibration
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if inference
 *  completed successfully.
 */
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR voice_run_classifier_continuous(
    voice_ei_impulse_handle_t *impulse,
    signal_t *signal,
    voice_ei_impulse_result_t *result,
    bool debug = false,
    bool enable_maf_unused = true)
{
    return voice_process_impulse_continuous(impulse, signal, result, debug);
}

/**
 * @brief Run the classifier over a raw features array.
 *
 *
 * Overloaded function [voice_run_classifier()](#voice_run_classifier-1) that defaults to the single impulse.
 *
 * **Blocking**: yes
 *
 * @param[in] signal Pointer to a `signal_t` struct that contains the total length of the raw
 *  feature array, which must match VOICE_EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, and a pointer to a callback
 *  that reads in the raw features.
 * @param[out] result  Pointer to an voice_ei_impulse_result_t struct that will contain the various output
 *  results from inference after `voice_run_classifier()` returns.
 * @param[in] debug Print internal preprocessing and inference debugging information via `voice_ei_printf()`.
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if inference
 *  completed successfully.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_run_classifier(
    signal_t *signal,
    voice_ei_impulse_result_t *result,
    bool debug = false)
{
    return voice_process_impulse(&voice_ei_default_impulse, signal, result, debug);
}

/**
 * @brief Run the classifier over a raw features array.
 *
 *
 * Accepts a `signal_t` input struct pointing to a callback that reads in pages of raw features.
 * `voice_run_classifier()` performs any necessary preprocessing on the raw features (e.g. DSP, cropping
 * of images, etc.) before performing inference. Results from inference are stored in an
 * `voice_ei_impulse_result_t` struct.
 *
 * **Blocking**: yes
 *
 * **Example**: [standalone inferencing main.cpp](https://github.com/edgeimpulse/example-standalone-inferencing/blob/master/source/main.cpp)
 *
 * @param[in] impulse Pointer to an `voice_ei_impulse_handle_t` struct that contains the model and
 *  preprocessing information.
 * @param[in] signal Pointer to a `signal_t` struct that contains the total length of the raw
 *  feature array, which must match VOICE_EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, and a pointer to a callback
 *  that reads in the raw features.
 * @param[out] result  Pointer to an voice_ei_impulse_result_t struct that will contain the various output
 *  results from inference after `voice_run_classifier()` returns.
 * @param[in] debug Print internal preprocessing and inference debugging information via `voice_ei_printf()`.
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if inference
 *  completed successfully.
 */
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR voice_run_classifier(
    voice_ei_impulse_handle_t *impulse,
    signal_t *signal,
    voice_ei_impulse_result_t *result,
    bool debug = false)
{
    return voice_process_impulse(impulse, signal, result, debug);
}

#if VOICE_EI_CLASSIFIER_FREEFORM_OUTPUT
/**
 * Set the location for freeform outputs. For impulses with freeform output the application needs to allocate
 * memory for all output tensors, and pass it to voice_ei_set_freeform_output. This memory is owned by the application.
 * Example usage:
 *
 * voice_ei_impulse_handle_t &impulse_handle = voice_ei_default_impulse;
 * std::vector<matrix_t> freeform_outputs;
 * freeform_outputs.reserve(impulse_handle.impulse->freeform_outputs_size);
 * for (size_t ix = 0; ix < impulse_handle.impulse->freeform_outputs_size; ++ix) {
 *     freeform_outputs.emplace_back(impulse_handle.impulse->freeform_outputs[ix], 1);
 * }
 *
 * int res = voice_ei_set_freeform_output(&impulse_handle, freeform_outputs.data(), freeform_outputs.size());
 * // Check that res == VOICE_EI_IMPULSE_OK
 *
 * @param[in] impulse_handle Pointer to an `voice_ei_impulse_handle_t` struct that contains the model and
 *  preprocessing information.
 * @param[in] freeform_outputs Pointer to array of ei::matrix structs that are sized according to the
 *  voice_ei_impulse_handle_t.impulse->freeform_outputs array.
 * @param[in] freeform_outputs_size Number of elements in freeform_outputs
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if setting the output
 *  was successful.
 */
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR voice_ei_set_freeform_output(
    voice_ei_impulse_handle_t *impulse_handle,
    ei::matrix_t *freeform_outputs,
    size_t freeform_outputs_size
) {
    // Check size of freeform_outputs_size
    if (freeform_outputs_size != impulse_handle->impulse->freeform_outputs_size) {
        VOICE_EI_LOGE("ERR: freeform_outputs_size should be of size %d, but was %d. You can get the required number of freeform outputs via impulse->freeform_outputs_size.\n",
            (int)freeform_outputs_size, (int)impulse_handle->impulse->freeform_outputs_size);
        return VOICE_EI_IMPULSE_FREEFORM_OUTPUT_SIZE_MISMATCH;
    }

    // Check size of each individual matrix
    for (size_t ix = 0; ix < freeform_outputs_size; ix++) {
        matrix_t& freeform_output = freeform_outputs[ix];
        if (freeform_output.rows * freeform_output.cols != impulse_handle->impulse->freeform_outputs[ix]) {
            VOICE_EI_LOGE("ERR: freeform_outputs at index %d has the wrong size. Expected %d elements, but freeform_output is %d elements. You can get the required size via impulse->freeform_outputs[%d].\n",
                (int)ix,
                (int)impulse_handle->impulse->freeform_outputs[ix],
                (int)freeform_output.rows * freeform_output.cols,
                (int)ix);
            return VOICE_EI_IMPULSE_FREEFORM_OUTPUT_SIZE_MISMATCH;
        }
    }

    impulse_handle->freeform_outputs = freeform_outputs;

    return VOICE_EI_IMPULSE_OK;
}

/**
 * @brief Set the location for freeform outputs. For impulses with freeform output the application needs to allocate
 * memory for all output tensors, and pass it to voice_ei_set_freeform_output. This memory is owned by the application.
 *
 * Overloaded function [voice_ei_set_freeform_output()](#voice_ei_set_freeform_output-0) that defaults to the default impulse.
 *
 * @param[in] freeform_outputs Pointer to array of ei::matrix structs that are sized according to the
 *  voice_ei_impulse_handle_t.impulse->freeform_outputs array.
 * @param[in] freeform_outputs_size Number of elements in freeform_outputs
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if setting the output
 *  was successful.
 */
extern "C" VOICE_EI_IMPULSE_ERROR voice_ei_set_freeform_output(
    ei::matrix_t *freeform_outputs,
    size_t freeform_outputs_size
) {
    return voice_ei_set_freeform_output(&voice_ei_default_impulse, freeform_outputs, freeform_outputs_size);
}
#endif // #if VOICE_EI_CLASSIFIER_FREEFORM_OUTPUT

/**
 * @brief Get image input parameters from an impulse
 *
 * @param handle voice_ei_impulse_handle_t
 * @param width uint32_t
 * @param height uint32_t
 * @param channels uint8_t
 *
 * @return VOICE_EI_IMPULSE_OK
 *
 * @brief This function retrieves the width, height, and channels of the input
 *     parameters from the given impulse. If the input parameters are not available,
 *     it returns the default values based on the impulse's input size.
 */
#if VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_VLM_CONNECTOR
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR voice_ei_get_image_input_params(
    voice_ei_impulse_handle_t *handle,
    uint32_t *width,
    uint32_t *height,
    uint8_t *channels
) {
    const voice_ei_impulse_t *impulse = handle->impulse;
    if (handle->input_params == nullptr) {
        *width = impulse->input_width;
        *height = impulse->input_height;
        *channels = impulse->nn_input_frame_size / (impulse->input_width * impulse->input_height);
    }
    else {
        *width = handle->input_params->input_width;
        *height = handle->input_params->input_height;
        *channels = handle->input_params->nn_input_frame_size / (handle->input_params->input_width * handle->input_params->input_height);
    }

    return VOICE_EI_IMPULSE_OK;
}

/**
 * @brief Set the image input parameters (width, height, and channels) for the given impulse handle.
 *
 * This function sets the dimensions and channel count of the input image for the given impulse handle.
 * It allocates and initializes a new `voice_ei_input_params` structure with the specified parameters.
 *
 * @param[in] handle Pointer to the impulse handle to update.
 * @param[in] width Width of the input image.
 * @param[in] height Height of the input image.
 * @param[in] channels Number of channels in the input image.
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Returns `VOICE_EI_IMPULSE_OK` if successful, or `VOICE_EI_IMPULSE_OUT_OF_MEMORY` if memory allocation fails.
 */
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR voice_ei_set_image_input_params(
    voice_ei_impulse_handle_t *handle,
    uint32_t width,
    uint32_t height,
    uint8_t channels
) {
    std::unique_ptr<voice_ei_input_params> params(new voice_ei_input_params());
    if (params == nullptr) {
        return VOICE_EI_IMPULSE_OUT_OF_MEMORY;
    }
    params->nn_input_frame_size = width * height * channels;
    params->raw_sample_count = width * height;
    params->raw_samples_per_frame = width * height;
    params->dsp_input_frame_size = width * height;
    params->input_width = width;
    params->input_height = height;
    params->input_frames = 1;
    params->interval_ms = 0.0f;
    params->frequency = 0.0f;

    handle->input_params = params.release();
    return VOICE_EI_IMPULSE_OK;
}
#endif // #if VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_VLM_CONNECTOR

/** @} */ // end of voice_ei_functions Doxygen group

/* Deprecated functions ------------------------------------------------------- */

/* These functions are being deprecated and possibly will be removed or moved in future.
Do not use these - if possible, change your code to reflect the upcoming changes. */

#if EIDSP_SIGNAL_C_FN_POINTER == 0

/**
 * @brief Run the impulse, if you provide an instance of sampler it will also persist
 *  the data for you.
 *
 * @deprecated This function is deprecated and will be removed in future versions. Use
 *  `voice_run_classifier()` instead.
 *
 * @param[in] sampler Instance to an **initialized** sampler
 * @param[out] result Object to store the results in
 * @param[in] data_fn Callback function to retrieve data from sensors
 * @param[in] debug Whether to log debug messages (default false)
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if inference
 *  completed successfully.
 */
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR run_impulse(
#if (defined(VOICE_EI_CLASSIFIER_HAS_SAMPLER) && VOICE_EI_CLASSIFIER_HAS_SAMPLER == 1) || defined(__DOXYGEN__)
        EdgeSampler *sampler,
#endif
        voice_ei_impulse_result_t *result,
#ifdef __MBED__
        mbed::Callback<void(float*, size_t)> data_fn,
#else
        std::function<void(float*, size_t)> data_fn,
#endif
        bool debug = false) {

    auto& impulse = *(voice_ei_default_impulse.impulse);

    float *x = (float*)calloc(impulse.dsp_input_frame_size, sizeof(float));
    if (!x) {
        return VOICE_EI_IMPULSE_OUT_OF_MEMORY;
    }

    uint64_t next_tick = 0;

    uint64_t sampling_us_start = voice_ei_read_timer_us();

    // grab some data
    for (int i = 0; i < (int)impulse.dsp_input_frame_size; i += impulse.raw_samples_per_frame) {
        uint64_t curr_us = voice_ei_read_timer_us() - sampling_us_start;

        next_tick = curr_us + (impulse.interval_ms * 1000);

        data_fn(x + i, impulse.raw_samples_per_frame);
#if defined(VOICE_EI_CLASSIFIER_HAS_SAMPLER) && VOICE_EI_CLASSIFIER_HAS_SAMPLER == 1
        if (sampler != NULL) {
            sampler->write_sensor_data(x + i, impulse.raw_samples_per_frame);
        }
#endif

        if (voice_ei_run_impulse_check_canceled() == VOICE_EI_IMPULSE_CANCELED) {
            free(x);
            return VOICE_EI_IMPULSE_CANCELED;
        }

        while (next_tick > voice_ei_read_timer_us() - sampling_us_start);
    }

    result->timing.sampling = (voice_ei_read_timer_us() - sampling_us_start) / 1000;

    signal_t signal;
    int err = numpy::signal_from_buffer(x, impulse.dsp_input_frame_size, &signal);
    if (err != 0) {
        free(x);
        voice_ei_printf("ERR: signal_from_buffer failed (%d)\n", err);
        return VOICE_EI_IMPULSE_DSP_ERROR;
    }

    VOICE_EI_IMPULSE_ERROR r = voice_run_classifier(&signal, result, debug);
    free(x);
    return r;
}

#if (defined(VOICE_EI_CLASSIFIER_HAS_SAMPLER) && VOICE_EI_CLASSIFIER_HAS_SAMPLER == 1) || defined(__DOXYGEN__)
/**
 * @brief Run the impulse, does not persist data.
 *
 * @deprecated This function is deprecated and will be removed in future versions. Use
 *  `voice_run_classifier()` instead.
 *
 * @param[out] result Object to store the results in
 * @param[in] data_fn Callback function to retrieve data from sensors
 * @param[out] debug Whether to log debug messages (default false)
 *
 * @return Error code as defined by `VOICE_EI_IMPULSE_ERROR` enum. Will be `VOICE_EI_IMPULSE_OK` if inference
 *  completed successfully.
 */
__attribute__((unused)) VOICE_EI_IMPULSE_ERROR run_impulse(
        voice_ei_impulse_result_t *result,
#ifdef __MBED__
        mbed::Callback<void(float*, size_t)> data_fn,
#else
        std::function<void(float*, size_t)> data_fn,
#endif
        bool debug = false) {
    return run_impulse(NULL, result, data_fn, debug);
}
#endif

#endif // #if EIDSP_SIGNAL_C_FN_POINTER == 0

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VOICE_EDGE_IMPULSE_RUN_CLASSIFIER_H_
