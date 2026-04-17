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

#ifndef _VOICE_EI_CLASSIFIER_INFERENCING_ENGINE_TFLITE_EON_H_
#define _VOICE_EI_CLASSIFIER_INFERENCING_ENGINE_TFLITE_EON_H_

#if (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE) && (VOICE_EI_CLASSIFIER_COMPILED == 1)

#include "voice-edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "voice-edge-impulse-sdk/tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_aligned_malloc.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_model_types.h"
#include "voice-edge-impulse-sdk/classifier/inferencing_engines/tflite_helper.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_run_dsp.h"

/**
 * Setup the TFLite runtime
 *
 * @param      ctx_start_us       Pointer to the start time
 * @param      input              Pointer to input tensor
 * @param      output             Pointer to output tensor
 * @param      micro_tensor_arena Pointer to the arena that will be allocated
 *
 * @return  VOICE_EI_IMPULSE_OK if successful
 */
static VOICE_EI_IMPULSE_ERROR inference_tflite_setup(
    voice_ei_learning_block_config_tflite_graph_t *block_config,
    uint64_t *ctx_start_us,
    TfLiteTensor* input,
    TfLiteTensor** output_arg,
    voice_ei_unique_ptr_t& p_tensor_arena) {

    *ctx_start_us = voice_ei_read_timer_us();

    TfLiteTensor *outputs = *output_arg;
    voice_ei_config_tflite_eon_graph_t *graph_config = (voice_ei_config_tflite_eon_graph_t*)block_config->graph_config;

    TfLiteStatus init_status = graph_config->model_init(voice_ei_aligned_calloc);
    if (init_status != kTfLiteOk) {
        voice_ei_printf("Failed to initialize the model (error code %d)\n", init_status);
        return VOICE_EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED;
    }

    TfLiteStatus status;

    status = graph_config->model_input(0, input);
    if (status != kTfLiteOk) {
        return VOICE_EI_IMPULSE_TFLITE_ERROR;
    }

    for (uint8_t i = 0; i < block_config->output_tensors_size; i++) {
        status = graph_config->model_output(block_config->output_tensors_indices[i], &outputs[i]);
        if (status != kTfLiteOk) {
            return VOICE_EI_IMPULSE_TFLITE_ERROR;
        }
    }

    return VOICE_EI_IMPULSE_OK;
}

/**
 * Run TFLite model
 *
 * @param   ctx_start_us    Start time of the setup function (see above)
 * @param   output          Output tensor
 * @param   interpreter     TFLite interpreter (non-compiled models)
 * @param   tensor_arena    Allocated arena (will be freed)
 * @param   result          Struct for results
 * @param   debug           Whether to print debug info
 *
 * @return  VOICE_EI_IMPULSE_OK if successful
 */
static VOICE_EI_IMPULSE_ERROR inference_tflite_run(
    const voice_ei_impulse_t *impulse,
    voice_ei_learning_block_config_tflite_graph_t *block_config,
    uint64_t ctx_start_us,
    TfLiteTensor** outputs,
    uint8_t* tensor_arena,
    voice_ei_impulse_result_t *result,
    bool debug) {

    voice_ei_config_tflite_eon_graph_t *graph_config = (voice_ei_config_tflite_eon_graph_t*)block_config->graph_config;

    if (graph_config->model_invoke() != kTfLiteOk) {
        return VOICE_EI_IMPULSE_TFLITE_ERROR;
    }

    uint64_t ctx_end_us = voice_ei_read_timer_us();

    result->timing.classification_us = ctx_end_us - ctx_start_us;

    VOICE_EI_LOGD("Predictions (time: %d ms.):\n", result->timing.classification);

    if (voice_ei_run_impulse_check_canceled() == VOICE_EI_IMPULSE_CANCELED) {
        return VOICE_EI_IMPULSE_CANCELED;
    }

    return VOICE_EI_IMPULSE_OK;
}

/**
 * @brief      Do neural network inferencing over a signal (from the DSP)
 *
 * @param      fmatrix  Processed matrix
 * @param      result   Output classifier results
 * @param[in]  debug    Debug output enable
 *
 * @return     The ei impulse error.
 */
VOICE_EI_IMPULSE_ERROR run_nn_inference_from_dsp(
    voice_ei_learning_block_config_tflite_graph_t *block_config,
    signal_t *signal,
    matrix_t *output_matrix)
{
    TfLiteTensor input;
    TfLiteTensor *outputs;

    // allocate outputs
    outputs = (TfLiteTensor*)voice_ei_malloc(block_config->output_tensors_size * sizeof(TfLiteTensor));

    uint64_t ctx_start_us = voice_ei_read_timer_us();
    voice_ei_unique_ptr_t p_tensor_arena(nullptr, voice_ei_aligned_free);
    voice_ei_config_tflite_eon_graph_t *graph_config = (voice_ei_config_tflite_eon_graph_t*)block_config->graph_config;

    VOICE_EI_IMPULSE_ERROR init_res = inference_tflite_setup(
        block_config,
        &ctx_start_us,
        &input,
        &outputs,
        p_tensor_arena);

    if (init_res != VOICE_EI_IMPULSE_OK) {
        return init_res;
    }

    auto input_res = fill_input_tensor_from_signal(signal, &input);
    if (input_res != VOICE_EI_IMPULSE_OK) {
        return input_res;
    }

    // invoke the model
    if (graph_config->model_invoke() != kTfLiteOk) {
        return VOICE_EI_IMPULSE_TFLITE_ERROR;
    }

    auto output_res = fill_output_matrix_from_tensor(&outputs[0], output_matrix);
    if (output_res != VOICE_EI_IMPULSE_OK) {
        return output_res;
    }

    if (graph_config->model_reset(voice_ei_aligned_free) != kTfLiteOk) {
        return VOICE_EI_IMPULSE_TFLITE_ERROR;
    }
    voice_ei_free(outputs);

    return VOICE_EI_IMPULSE_OK;
}

/**
 * @brief      Do neural network inferencing over a feature matrix
 *
 * @param      fmatrix  Processed matrix
 * @param      result   Output classifier results
 * @param[in]  debug    Debug output enable
 *
 * @return     The ei impulse error.
 */
VOICE_EI_IMPULSE_ERROR run_nn_inference(
    const voice_ei_impulse_t *impulse,
    voice_ei_feature_t *fmatrix,
    uint32_t learn_block_index,
    uint32_t* input_block_ids,
    uint32_t input_block_ids_size,
    voice_ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false)
{
    voice_ei_learning_block_config_tflite_graph_t *block_config = (voice_ei_learning_block_config_tflite_graph_t*)config_ptr;
    voice_ei_config_tflite_eon_graph_t *graph_config = (voice_ei_config_tflite_eon_graph_t*)block_config->graph_config;

    TfLiteTensor input;
    TfLiteTensor *outputs;

    // allocate outputs
    outputs = (TfLiteTensor*)voice_ei_malloc(block_config->output_tensors_size * sizeof(TfLiteTensor));

    uint64_t ctx_start_us = voice_ei_read_timer_us();
    voice_ei_unique_ptr_t p_tensor_arena(nullptr, voice_ei_aligned_free);

    VOICE_EI_IMPULSE_ERROR init_res = inference_tflite_setup(
        block_config,
        &ctx_start_us,
        &input,
        &outputs,
        p_tensor_arena);

    if (init_res != VOICE_EI_IMPULSE_OK) {
        return init_res;
    }

    uint8_t* tensor_arena = static_cast<uint8_t*>(p_tensor_arena.get());

    auto input_res = fill_input_tensor_from_matrix(fmatrix,
                                                   result->_raw_outputs,
                                                   &input,
                                                   input_block_ids,
                                                   input_block_ids_size,
                                                   impulse->dsp_blocks_size,
                                                   impulse->learning_blocks_size);

    if (input_res != VOICE_EI_IMPULSE_OK) {
        return input_res;
    }

    VOICE_EI_IMPULSE_ERROR run_res = inference_tflite_run(
        impulse,
        block_config,
        ctx_start_us,
        &outputs,
        tensor_arena, result, debug);

    for (uint32_t output_ix = 0; output_ix < block_config->output_tensors_size; output_ix++) {
        TfLiteTensor* output = &outputs[output_ix];
        // calculate the size of the output by iterating through dims
        size_t output_size = 1;
        for (int dim_num = 0; dim_num < output->dims->size; dim_num++) {
            output_size *= output->dims->data[dim_num];
        }
        switch (output->type) {
            case kTfLiteFloat32: {
                result->_raw_outputs[learn_block_index + output_ix].matrix = new matrix_t(1, output_size);
                memcpy(result->_raw_outputs[learn_block_index + output_ix].matrix->buffer, output->data.f, output->bytes);
                break;
            }
            case kTfLiteInt8: {
                if (block_config->dequantize_output) {
                    result->_raw_outputs[learn_block_index + output_ix].matrix = new matrix_t(1, output_size);
                    fill_output_matrix_from_tensor(output, result->_raw_outputs[learn_block_index + output_ix].matrix);
                }
                else {
                    result->_raw_outputs[learn_block_index + output_ix].matrix_i8 = new matrix_i8_t(1, output_size);
                    memcpy(result->_raw_outputs[learn_block_index + output_ix].matrix_i8->buffer, output->data.int8, output->bytes);
                }
                break;
            }
            case kTfLiteUInt8: {
                if (block_config->dequantize_output) {
                    result->_raw_outputs[learn_block_index + output_ix].matrix = new matrix_t(1, output_size);
                    fill_output_matrix_from_tensor(output, result->_raw_outputs[learn_block_index + output_ix].matrix);
                }
                else {
                    result->_raw_outputs[learn_block_index + output_ix].matrix_u8 = new matrix_u8_t(1, output_size);
                    memcpy(result->_raw_outputs[learn_block_index + output_ix].matrix_u8->buffer, output->data.uint8, output->bytes);
                }
                break;
            }
            default: {
                voice_ei_printf("ERR: Cannot handle output type (%d)\n", output->type);
                return VOICE_EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL;
            }
        }

        result->_raw_outputs[learn_block_index + output_ix].blockId = block_config->block_id + output_ix;
    }

    graph_config->model_reset(voice_ei_aligned_free);
    voice_ei_free(outputs);

    if (run_res != VOICE_EI_IMPULSE_OK) {
        return run_res;
    }

    return VOICE_EI_IMPULSE_OK;
}

#if VOICE_EI_CLASSIFIER_QUANTIZATION_ENABLED == 1
/**
 * Special function to run the classifier on images, only works on TFLite models (either interpreter or EON or for tensaiflow)
 * that allocates a lot less memory by quantizing in place. This only works if 'can_voice_run_classifier_image_quantized'
 * returns VOICE_EI_IMPULSE_OK.
 */
VOICE_EI_IMPULSE_ERROR run_nn_inference_image_quantized(
    const voice_ei_impulse_t *impulse,
    signal_t *signal,
    uint32_t learn_block_index,
    voice_ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false) {

    voice_ei_learning_block_config_tflite_graph_t *block_config = (voice_ei_learning_block_config_tflite_graph_t*)config_ptr;
    voice_ei_config_tflite_eon_graph_t *graph_config = (voice_ei_config_tflite_eon_graph_t*)block_config->graph_config;

    uint64_t ctx_start_us;
    TfLiteTensor input;
    TfLiteTensor *outputs;

    // allocate outputs
    outputs = (TfLiteTensor*)voice_ei_malloc(block_config->output_tensors_size * sizeof(TfLiteTensor));

    voice_ei_unique_ptr_t p_tensor_arena(nullptr, voice_ei_aligned_free);

    VOICE_EI_IMPULSE_ERROR init_res = inference_tflite_setup(
        block_config,
        &ctx_start_us,
        &input,
        &outputs,
        p_tensor_arena);

    if (init_res != VOICE_EI_IMPULSE_OK) {
        return init_res;
    }

    if (input.type != TfLiteType::kTfLiteInt8 && input.type != TfLiteType::kTfLiteUInt8) {
        return VOICE_EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
    }

    uint64_t dsp_start_us = voice_ei_read_timer_us();

    // features matrix maps around the input tensor to not allocate any memory
    ei::matrix_i8_t features_matrix(1, impulse->nn_input_frame_size, input.data.int8);

    // run DSP process and quantize automatically
    int ret = extract_image_features_quantized(signal, &features_matrix, impulse->dsp_blocks[0].config, input.params.scale, input.params.zero_point,
        impulse->frequency, impulse->learning_blocks[0].image_scaling);

    if (ret != EIDSP_OK) {
        voice_ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
        return VOICE_EI_IMPULSE_DSP_ERROR;
    }

    if (voice_ei_run_impulse_check_canceled() == VOICE_EI_IMPULSE_CANCELED) {
        return VOICE_EI_IMPULSE_CANCELED;
    }

    result->timing.dsp_us = voice_ei_read_timer_us() - dsp_start_us;

    if (debug) {
        voice_ei_printf("Features (%d ms.): ", result->timing.dsp);
        for (size_t ix = 0; ix < features_matrix.cols; ix++) {
            voice_ei_printf_float((features_matrix.buffer[ix] - input.params.zero_point) * input.params.scale);
            voice_ei_printf(" ");
        }
        voice_ei_printf("\n");
    }

    ctx_start_us = voice_ei_read_timer_us();

    VOICE_EI_IMPULSE_ERROR run_res = inference_tflite_run(
        impulse,
        block_config,
        ctx_start_us,
        &outputs,
        static_cast<uint8_t*>(p_tensor_arena.get()),
        result,
        debug);

    for (uint32_t output_ix = 0; output_ix < block_config->output_tensors_size; output_ix++) {
        TfLiteTensor* output = &outputs[output_ix];
        // calculate the size of the output by iterating through dims
        size_t output_size = 1;
        for (int dim_num = 0; dim_num < output->dims->size; dim_num++) {
            output_size *= output->dims->data[dim_num];
        }

        switch (output->type) {
            case kTfLiteFloat32: {
                result->_raw_outputs[learn_block_index + output_ix].matrix = new matrix_t(1, output_size);
                memcpy(result->_raw_outputs[learn_block_index + output_ix].matrix->buffer, output->data.f, output->bytes);
                break;
            }
            case kTfLiteInt8: {
                if (block_config->dequantize_output) {
                    result->_raw_outputs[learn_block_index + output_ix].matrix = new matrix_t(1, output_size);
                    fill_output_matrix_from_tensor(output, result->_raw_outputs[learn_block_index + output_ix].matrix);
                }
                else {
                    result->_raw_outputs[learn_block_index + output_ix].matrix_i8 = new matrix_i8_t(1, output_size);
                    memcpy(result->_raw_outputs[learn_block_index + output_ix].matrix_i8->buffer, output->data.int8, output->bytes);
                }
                break;
            }
            case kTfLiteUInt8: {
                if (block_config->dequantize_output) {
                    result->_raw_outputs[learn_block_index + output_ix].matrix = new matrix_t(1, output_size);
                    fill_output_matrix_from_tensor(output, result->_raw_outputs[learn_block_index + output_ix].matrix);
                }
                else {
                    result->_raw_outputs[learn_block_index + output_ix].matrix_u8 = new matrix_u8_t(1, output_size);
                    memcpy(result->_raw_outputs[learn_block_index + output_ix].matrix_u8->buffer, output->data.uint8, output->bytes);
                }
                break;
            }
            default: {
                voice_ei_printf("ERR: Cannot handle output type (%d)\n", output->type);
                return VOICE_EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL;
            }
        }

        result->_raw_outputs[learn_block_index + output_ix].blockId = block_config->block_id + output_ix;
    }

    graph_config->model_reset(voice_ei_aligned_free);
    voice_ei_free(outputs);

    if (run_res != VOICE_EI_IMPULSE_OK) {
        return run_res;
    }

    return VOICE_EI_IMPULSE_OK;
}
#endif // VOICE_EI_CLASSIFIER_QUANTIZATION_ENABLED == 1

__attribute__((unused)) int extract_tflite_eon_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float frequency) {
    voice_ei_dsp_config_tflite_eon_t *dsp_config = (voice_ei_dsp_config_tflite_eon_t*)config_ptr;

    voice_ei_config_tflite_eon_graph_t voice_ei_config_tflite_graph_0 = {
        .implementation_version = 1,
        .model_init = dsp_config->init_fn,
        .model_invoke = dsp_config->invoke_fn,
        .model_reset = dsp_config->reset_fn,
        .model_input = dsp_config->input_fn,
        .model_output = dsp_config->output_fn,
    };

    const uint8_t voice_ei_output_tensor_indices[1] = { 0 };
    const uint8_t voice_ei_output_tensor_size = 1;

    voice_ei_learning_block_config_tflite_graph_t voice_ei_learning_block_config = {
        .implementation_version = 1,
        .block_id = dsp_config->block_id,
        .output_tensors_indices = voice_ei_output_tensor_indices,
        .output_tensors_size = voice_ei_output_tensor_size,
        .quantized = 0,
        .compiled = 1,
        .graph_config = &voice_ei_config_tflite_graph_0
    };

    auto x = run_nn_inference_from_dsp(&voice_ei_learning_block_config, signal, output_matrix);
    if (x != 0) {
        return x;
    }

    return EIDSP_OK;
}

#endif // (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_TFLITE) && (VOICE_EI_CLASSIFIER_COMPILED == 1)
#endif // _VOICE_EI_CLASSIFIER_INFERENCING_ENGINE_TFLITE_EON_H_
