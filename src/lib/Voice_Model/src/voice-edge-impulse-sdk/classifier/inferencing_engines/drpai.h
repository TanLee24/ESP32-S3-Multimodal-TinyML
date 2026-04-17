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

#ifndef _VOICE_EI_CLASSIFIER_INFERENCING_ENGINE_DRPAI_H_
#define _VOICE_EI_CLASSIFIER_INFERENCING_ENGINE_DRPAI_H_

#if (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI)

/*****************************************
 * includes
 ******************************************/
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <voice-model-parameters/model_metadata.h>

#if ((VOICE_EI_CLASSIFIER_OBJECT_DETECTION == 1) && (VOICE_EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER == VOICE_EI_CLASSIFIER_LAST_LAYER_YOLOV5_V5_DRPAI))
// For a YOLOV5_V5_DRPAI model we ran the unsupported layers with TF
#include <thread>
#include "voice-edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "voice-edge-impulse-sdk/tensorflow/lite/interpreter.h"
#include "voice-edge-impulse-sdk/tensorflow/lite/kernels/register.h"
#include "voice-edge-impulse-sdk/tensorflow/lite/model.h"
#include "voice-edge-impulse-sdk/tensorflow/lite/optional_debug_tools.h"
#endif
#include "voice-edge-impulse-sdk/tensorflow/lite/kernels/custom/tree_ensemble_classifier.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_model_types.h"
#include "voice-edge-impulse-sdk/classifier/voice_ei_run_dsp.h"
#include "voice-edge-impulse-sdk/porting/voice_ei_logging.h"

#include <linux/drpai.h>
#include <voice-tflite-model/drpai_model.h>

/*****************************************
 * Macro
 ******************************************/
/*Maximum DRP-AI Timeout threshold*/
#define DRPAI_TIMEOUT (5)

/*Buffer size for writing data to memory via DRP-AI Driver.*/
#define BUF_SIZE (1024)

/*Index to access drpai_file_path[]*/
#define INDEX_D (0)
#define INDEX_C (1)
#define INDEX_P (2)
#define INDEX_A (3)
#define INDEX_W (4)

/*****************************************
 * Public global vars
 ******************************************/
// input and output buffer pointers for memory mapped regions used by DRP-AI
uint8_t *drpai_input_buf = (uint8_t *)NULL;
float *drpai_output_buf = (float *)NULL;

/*****************************************
 * Typedef
 ******************************************/
/* For DRP-AI Address List */
typedef struct {
  unsigned long desc_aimac_addr;
  unsigned long desc_aimac_size;
  unsigned long desc_drp_addr;
  unsigned long desc_drp_size;
  unsigned long drp_param_addr;
  unsigned long drp_param_size;
  unsigned long data_in_addr;
  unsigned long data_in_size;
  unsigned long data_addr;
  unsigned long data_size;
  unsigned long work_addr;
  unsigned long work_size;
  unsigned long data_out_addr;
  unsigned long data_out_size;
  unsigned long drp_config_addr;
  unsigned long drp_config_size;
  unsigned long weight_addr;
  unsigned long weight_size;
} st_addr_t;

/*****************************************
 * static vars
 ******************************************/
static st_addr_t drpai_address;
static uint64_t udmabuf_address = 0;

static int drpai_fd = -1;

drpai_data_t proc[DRPAI_INDEX_NUM];

void get_udmabuf_memory_start_addr()
{   /* Obtain udmabuf memory area starting address */

    int8_t fd = 0;
    char addr[1024];
    int32_t read_ret = 0;
    errno = 0;

    fd = open("/sys/class/u-dma-buf/udmabuf0/phys_addr", O_RDONLY);
    if (0 > fd)
    {
        fprintf(stderr, "[ERROR] Failed to open udmabuf0/phys_addr : errno=%d\n", errno);
    }

    read_ret = read(fd, addr, 1024);
    if (0 > read_ret)
    {
        fprintf(stderr, "[ERROR] Failed to read udmabuf0/phys_addr : errno=%d\n", errno);
        close(fd);
    }

    sscanf(addr, "%lx", &udmabuf_address);
    close(fd);

    /* Filter the bit higher than 32 bit */
    udmabuf_address &=0xFFFFFFFF;
}

uint8_t drpai_init_mem(uint32_t input_frame_size) {
  int32_t i = 0;

  int udmabuf_fd0 = open("/dev/udmabuf0", O_RDWR);
  if (udmabuf_fd0 < 0) {
    return -1;
  }

  // input_frame_size === data_in_size
  uint8_t *addr =
      (uint8_t *)mmap(NULL, input_frame_size,
                      PROT_READ | PROT_WRITE, MAP_SHARED, udmabuf_fd0, 0);

  drpai_input_buf = addr;

  /* Write once to allocate physical memory to u-dma-buf virtual space.
   * Note: Do not use memset() for this.
   *       Because it does not work as expected. */
  for (i = 0; i < input_frame_size; i++) {
    drpai_input_buf[i] = 0;
  }


  get_udmabuf_memory_start_addr();
  if (0 == udmabuf_address) {
    return VOICE_EI_IMPULSE_DRPAI_INIT_FAILED;
  }

  return 0;
}

/*****************************************
 * Function Name : read_addrmap_txt
 * Description	: Loads address and size of DRP-AI Object files into struct
 *addr. Arguments		: addr_file = filename of addressmap file (from
 *DRP-AI Object files) Return value	: 0 if succeeded not 0 otherwise
 ******************************************/
static int8_t read_addrmap_txt() {
  // create a stream from the DRP-AI model data without copying
  std::istringstream ifs;
  ifs.rdbuf()->pubsetbuf((char *)voice_ei_voice_ei_addrmap_intm_txt, voice_ei_voice_ei_addrmap_intm_txt_len);

  std::string str;
  unsigned long l_addr;
  unsigned long l_size;
  std::string element, a, s;

  if (ifs.fail()) {
    return -1;
  }

  while (getline(ifs, str)) {
    std::istringstream iss(str);
    iss >> element >> a >> s;
    l_addr = strtol(a.c_str(), NULL, 16);
    l_size = strtol(s.c_str(), NULL, 16);

    if (element == "drp_config") {
      drpai_address.drp_config_addr = l_addr;
      drpai_address.drp_config_size = l_size;
    } else if (element == "desc_aimac") {
      drpai_address.desc_aimac_addr = l_addr;
      drpai_address.desc_aimac_size = l_size;
    } else if (element == "desc_drp") {
      drpai_address.desc_drp_addr = l_addr;
      drpai_address.desc_drp_size = l_size;
    } else if (element == "drp_param") {
      drpai_address.drp_param_addr = l_addr;
      drpai_address.drp_param_size = l_size;
    } else if (element == "weight") {
      drpai_address.weight_addr = l_addr;
      drpai_address.weight_size = l_size;
    } else if (element == "data_in") {
      drpai_address.data_in_addr = l_addr;
      drpai_address.data_in_size = l_size;
    } else if (element == "data") {
      drpai_address.data_addr = l_addr;
      drpai_address.data_size = l_size;
    } else if (element == "data_out") {
      drpai_address.data_out_addr = l_addr;
      drpai_address.data_out_size = l_size;
    } else if (element == "work") {
      drpai_address.work_addr = l_addr;
      drpai_address.work_size = l_size;
    }
  }

  return 0;
}

/*****************************************
 * Function Name : load_data_to_mem
 * Description	: Loads a binary blob DRP-AI Driver Memory
 * Arguments		: data_ptr = pointer to the bytes to write
 *				  drpai_fd = file descriptor of DRP-AI Driver
 *				  from = memory start address where the data is
 *written size = data size to be written Return value	: 0 if succeeded not 0
 *otherwise
 ******************************************/
static int8_t load_data_to_mem(unsigned char *data_ptr, int drpai_fd,
                               unsigned long from, unsigned long size) {
  drpai_data_t drpai_data;

  drpai_data.address = from;
  drpai_data.size = size;

  errno = 0;
  if (-1 == ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data)) {
    return -1;
  }

  errno = 0;
  if (-1 == write(drpai_fd, data_ptr, size)) {
    return -1;
  }

  return 0;
}

/*****************************************
 * Function Name :  load_drpai_data
 * Description	: Loads DRP-AI Object files to memory via DRP-AI Driver.
 * Arguments		: drpai_fd = file descriptor of DRP-AI Driver
 * Return value	: 0 if succeeded
 *				: not 0 otherwise
 ******************************************/
static int load_drpai_data(int drpai_fd) {
  unsigned long addr, size;
  unsigned char *data_ptr;
  for (int i = 0; i < 5; i++) {
    switch (i) {
    case (INDEX_W):
      addr = drpai_address.weight_addr;
      size = drpai_address.weight_size;
      data_ptr = voice_ei_voice_ei_weight_dat;
      break;
    case (INDEX_C):
      addr = drpai_address.drp_config_addr;
      size = drpai_address.drp_config_size;
      data_ptr = voice_ei_voice_ei_drpcfg_mem;
      break;
    case (INDEX_P):
      addr = drpai_address.drp_param_addr;
      size = drpai_address.drp_param_size;
      data_ptr = voice_ei_drp_param_bin;
      break;
    case (INDEX_A):
      addr = drpai_address.desc_aimac_addr;
      size = drpai_address.desc_aimac_size;
      data_ptr = voice_ei_aimac_desc_bin;
      break;
    case (INDEX_D):
      addr = drpai_address.desc_drp_addr;
      size = drpai_address.desc_drp_size;
      data_ptr = voice_ei_drp_desc_bin;
      break;
    default:
      return -1;
      break;
    }
    if (0 != load_data_to_mem(data_ptr, drpai_fd, addr, size)) {
      return -1;
    }
  }
  return 0;
}

VOICE_EI_IMPULSE_ERROR drpai_init_classifier() {
  // retval for drpai status
  int ret_drpai;

  // Read DRP-AI Object files address and size
  if (0 != read_addrmap_txt()) {
    voice_ei_printf("ERR: read_addrmap_txt failed : %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_INIT_FAILED;
  }

  // DRP-AI Driver Open
  drpai_fd = open("/dev/drpai0", O_RDWR);
  if (drpai_fd < 0) {
    voice_ei_printf("ERR: Failed to Open DRP-AI Driver: errno=%d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_INIT_FAILED;
  }

  // Load DRP-AI Data from Filesystem to Memory via DRP-AI Driver
  ret_drpai = load_drpai_data(drpai_fd);
  if (ret_drpai != 0) {
    voice_ei_printf("ERR: Failed to load DRPAI Data\n");
    if (0 != close(drpai_fd)) {
      voice_ei_printf("ERR: Failed to Close DRPAI Driver: errno=%d\n", errno);
    }
    return VOICE_EI_IMPULSE_DRPAI_INIT_FAILED;
  }

  // statically store DRP object file addresses and sizes
  proc[DRPAI_INDEX_INPUT].address = (uint32_t)udmabuf_address;
  proc[DRPAI_INDEX_INPUT].size = drpai_address.data_in_size;
  proc[DRPAI_INDEX_DRP_CFG].address = drpai_address.drp_config_addr;
  proc[DRPAI_INDEX_DRP_CFG].size = drpai_address.drp_config_size;
  proc[DRPAI_INDEX_DRP_PARAM].address = drpai_address.drp_param_addr;
  proc[DRPAI_INDEX_DRP_PARAM].size = drpai_address.drp_param_size;
  proc[DRPAI_INDEX_AIMAC_DESC].address = drpai_address.desc_aimac_addr;
  proc[DRPAI_INDEX_AIMAC_DESC].size = drpai_address.desc_aimac_size;
  proc[DRPAI_INDEX_DRP_DESC].address = drpai_address.desc_drp_addr;
  proc[DRPAI_INDEX_DRP_DESC].size = drpai_address.desc_drp_size;
  proc[DRPAI_INDEX_WEIGHT].address = drpai_address.weight_addr;
  proc[DRPAI_INDEX_WEIGHT].size = drpai_address.weight_size;
  proc[DRPAI_INDEX_OUTPUT].address = drpai_address.data_out_addr;
  proc[DRPAI_INDEX_OUTPUT].size = drpai_address.data_out_size;

  VOICE_EI_LOGD("proc[DRPAI_INDEX_INPUT] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_INPUT].address, proc[DRPAI_INDEX_INPUT].size);
  VOICE_EI_LOGD("proc[DRPAI_INDEX_DRP_CFG] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_DRP_CFG].address, proc[DRPAI_INDEX_DRP_CFG].size);
  VOICE_EI_LOGD("proc[DRPAI_INDEX_DRP_PARAM] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_DRP_PARAM].address, proc[DRPAI_INDEX_DRP_PARAM].size);
  VOICE_EI_LOGD("proc[DRPAI_INDEX_AIMAC_DESC] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_AIMAC_DESC].address, proc[DRPAI_INDEX_AIMAC_DESC].size);
  VOICE_EI_LOGD("proc[DRPAI_INDEX_DRP_DESC] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_DRP_DESC].address, proc[DRPAI_INDEX_DRP_DESC].size);
  VOICE_EI_LOGD("proc[DRPAI_INDEX_WEIGHT] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_WEIGHT].address, proc[DRPAI_INDEX_WEIGHT].size);
  VOICE_EI_LOGD("proc[DRPAI_INDEX_OUTPUT] addr: %p, size: %p\r\n", proc[DRPAI_INDEX_OUTPUT].address, proc[DRPAI_INDEX_OUTPUT].size);

  drpai_output_buf = (float *)voice_ei_malloc(drpai_address.data_out_size);

  return VOICE_EI_IMPULSE_OK;
}

VOICE_EI_IMPULSE_ERROR drpai_voice_run_classifier_image_quantized() {
#if VOICE_EI_CLASSIFIER_COMPILED == 1
#error "DRP-AI is not compatible with EON Compiler"
#endif
  // output data from DRPAI model
  drpai_data_t drpai_data;
  // status used to query if any internal errors occured during inferencing
  drpai_status_t drpai_status;
  // descriptor used for checking if DRPAI is done inferencing
  fd_set rfds;
  // struct used to define DRPAI timeout
  struct timespec tv;
  // retval for drpai status
  int ret_drpai;
  // retval when querying drpai status
  int inf_status = 0;

  // DRP-AI Output Memory Preparation
  drpai_data.address = drpai_address.data_out_addr;
  drpai_data.size = drpai_address.data_out_size;

  // Start DRP-AI driver
  VOICE_EI_LOGD("Start DRPAI inference\r\n");
  int ioret = ioctl(drpai_fd, DRPAI_START, &proc[0]);
  if (0 != ioret) {
    VOICE_EI_LOGE("Failed to Start DRPAI Inference: %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
  }

  // Settings For pselect - this is how DRPAI signals inferencing complete
  FD_ZERO(&rfds);
  FD_SET(drpai_fd, &rfds);
  // Define a timeout for DRP-AI to complete
  tv.tv_sec = DRPAI_TIMEOUT;
  tv.tv_nsec = 0;

  // Wait until DRP-AI ends
  VOICE_EI_LOGD("Waiting on DRPAI inference results\r\n");
  ret_drpai = pselect(drpai_fd + 1, &rfds, NULL, NULL, &tv, NULL);
  if (ret_drpai == 0) {
      VOICE_EI_LOGE("DRPAI Inference pselect() Timeout: %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
  } else if (ret_drpai < 0) {
      VOICE_EI_LOGE("DRPAI Inference pselect() Error: %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
  }

  // Checks for DRPAI inference status errors
  VOICE_EI_LOGD("Getting DRPAI Status\r\n");
  inf_status = ioctl(drpai_fd, DRPAI_GET_STATUS, &drpai_status);
  if (inf_status != 0) {
      VOICE_EI_LOGE("DRPAI Internal Error: %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
  }

  VOICE_EI_LOGD("Getting inference results\r\n");
  if (ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data) != 0) {
      VOICE_EI_LOGE("Failed to Assign DRPAI data: %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
  }

  if (read(drpai_fd, drpai_output_buf, drpai_data.size) < 0) {
      VOICE_EI_LOGE("Failed to read DRPAI output data: %d\n", errno);
    return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
  }
  return VOICE_EI_IMPULSE_OK;
}

// close the driver (reset file handles)
VOICE_EI_IMPULSE_ERROR drpai_close(uint32_t input_frame_size) {
  munmap(drpai_input_buf, input_frame_size);
  free(drpai_output_buf);
  if (drpai_fd > 0) {
    if (0 != close(drpai_fd)) {
        VOICE_EI_LOGE("Failed to Close DRP-AI Driver: errno=%d\n", errno);
      return VOICE_EI_IMPULSE_DRPAI_RUNTIME_FAILED;
    }
    drpai_fd = -1;
  }
  return VOICE_EI_IMPULSE_OK;
}

#if ((VOICE_EI_CLASSIFIER_OBJECT_DETECTION == 1) && (VOICE_EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER == VOICE_EI_CLASSIFIER_LAST_LAYER_YOLOV5_V5_DRPAI))
VOICE_EI_IMPULSE_ERROR drpai_run_yolov5_postprocessing(
    const voice_ei_impulse_t *impulse,
    float** output)
{
    static std::unique_ptr<tflite::FlatBufferModel> model = nullptr;
    static std::unique_ptr<tflite::Interpreter> interpreter = nullptr;

    if (!model) {
        model = tflite::FlatBufferModel::BuildFromBuffer((const char*)yolov5_part2, yolov5_part2_len);
        if (!model) {
            voice_ei_printf("Failed to build TFLite model from buffer\n");
            return VOICE_EI_IMPULSE_TFLITE_ERROR;
        }

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);

        if (!interpreter) {
            voice_ei_printf("Failed to construct interpreter\n");
            return VOICE_EI_IMPULSE_TFLITE_ERROR;
        }

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            voice_ei_printf("AllocateTensors failed\n");
            return VOICE_EI_IMPULSE_TFLITE_ERROR;
        }

        int hw_thread_count = (int)std::thread::hardware_concurrency();
        hw_thread_count -= 1; // leave one thread free for the other application
        if (hw_thread_count < 1) {
            hw_thread_count = 1;
        }

        if (interpreter->SetNumThreads(hw_thread_count) != kTfLiteOk) {
            voice_ei_printf("SetNumThreads failed\n");
            return VOICE_EI_IMPULSE_TFLITE_ERROR;
        }
    }

    const size_t drpai_buff_size = drpai_address.data_out_size / sizeof(float);
    const size_t drpai_features = drpai_buff_size;

    const size_t els_per_grid = drpai_features / ((NUM_GRID_1 * NUM_GRID_1) + (NUM_GRID_2 * NUM_GRID_2) + (NUM_GRID_3 * NUM_GRID_3));

    const size_t grid_1_offset = 0;
    const size_t grid_1_size = (NUM_GRID_1 * NUM_GRID_1) * els_per_grid;

    const size_t grid_2_offset = grid_1_offset + grid_1_size;
    const size_t grid_2_size = (NUM_GRID_2 * NUM_GRID_2) * els_per_grid;

    const size_t grid_3_offset = grid_2_offset + grid_2_size;
    const size_t grid_3_size = (NUM_GRID_3 * NUM_GRID_3) * els_per_grid;

    // Now we don't know the exact tensor order for some reason
    // so let's do that dynamically
    for (size_t ix = 0; ix < 3; ix++) {
        TfLiteTensor * tensor = interpreter->input_tensor(ix);
        size_t tensor_size = 1;
        for (size_t ix = 0; ix < tensor->dims->size; ix++) {
            tensor_size *= tensor->dims->data[ix];
        }

        VOICE_EI_LOGD("input tensor %d, tensor_size=%d\n", (int)ix, (int)tensor_size);

        float *input = interpreter->typed_input_tensor<float>(ix);

        if (tensor_size == grid_1_size) {
            memcpy(input, drpai_output_buf + grid_1_offset, grid_1_size * sizeof(float));
        }
        else if (tensor_size == grid_2_size) {
            memcpy(input, drpai_output_buf + grid_2_offset, grid_2_size * sizeof(float));
        }
        else if (tensor_size == grid_3_size) {
            memcpy(input, drpai_output_buf + grid_3_offset, grid_3_size * sizeof(float));
        }
        else {
            voice_ei_printf("ERR: Cannot determine which grid to use for input tensor %d with %d tensor size\n",
                (int)ix, (int)tensor_size);
            return VOICE_EI_IMPULSE_TFLITE_ERROR;
        }
    }

    uint64_t ctx_start_us = voice_ei_read_timer_us();
    interpreter->Invoke();
    uint64_t ctx_end_us = voice_ei_read_timer_us();
    VOICE_EI_LOGD("Invoke took %d ms.\n", (int)((ctx_end_us - ctx_start_us) / 1000));

    *output = interpreter->typed_output_tensor<float>(0);

    return VOICE_EI_IMPULSE_OK;
}
#endif

/**
 * @brief      Do neural network inferencing over the processed feature matrix
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
    bool debug)
{
    // dummy, not used for DRPAI
}

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
    bool debug = false)
{
  voice_ei_learning_block_config_tflite_graph_t *block_config = (voice_ei_learning_block_config_tflite_graph_t*)config_ptr;
  voice_ei_config_drpai_graph_t *graph_config = (voice_ei_config_drpai_graph_t*)block_config->graph_config;

    // this needs to be changed for multi-model, multi-impulse
    static bool first_run = true;
    uint64_t ctx_start_us;
    uint64_t dsp_start_us = voice_ei_read_timer_us();

    if (first_run) {
        // map memory regions to the DRP-AI UDMA. This is required for passing data
        // to and from DRP-AI
        int t = drpai_init_mem(impulse->nn_input_frame_size);
        if (t != 0) {
            return VOICE_EI_IMPULSE_DRPAI_INIT_FAILED;
        }

        VOICE_EI_IMPULSE_ERROR ret = drpai_init_classifier();
        if (ret != VOICE_EI_IMPULSE_OK) {
            drpai_close(impulse->nn_input_frame_size);
            return VOICE_EI_IMPULSE_DRPAI_INIT_FAILED;
        }

        VOICE_EI_LOGI("Initialized input and output buffers:\r\n");
        VOICE_EI_LOGI("input buf (addr: %p, size: 0x%x)\r\n", drpai_input_buf, drpai_address.data_in_size);
        VOICE_EI_LOGI("output buf (addr: %p, size: 0x%x)\r\n", drpai_output_buf, drpai_address.data_out_size);
        VOICE_EI_LOGI("udmabuf_addr: %p\n", udmabuf_address);
    }

    VOICE_EI_LOGD("Starting DSP...\n");
    int ret;

    VOICE_EI_LOGD("fmatrix size == Bpp * signal.total_length ( %p == %p * %p = %p )\r\n", proc[DRPAI_INDEX_INPUT].size, 3, signal->total_length, 3 * signal->total_length);
    // Creates a features matrix mapped to the DRP-AI UDMA input region
    ei::matrix_u8_t features_matrix(1, proc[DRPAI_INDEX_INPUT].size, drpai_input_buf);

    // Grabs the raw image buffer from the signal, DRP-AI will automatically
    // extract features
    ret = extract_drpai_features_quantized(
        signal,
        &features_matrix,
        impulse->dsp_blocks[0].config,
        impulse->frequency);
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
      for (size_t ix = 0; ix < VOICE_EI_CLASSIFIER_NN_INPUT_FRAME_SIZE; ix++) {
          voice_ei_printf("0x%hhx, ", drpai_input_buf[ix]);
      }
      voice_ei_printf("\n");
    }

    ctx_start_us = voice_ei_read_timer_us();

    // Run DRP-AI inference, a static buffer is used to store the raw output
    // results
    ret = drpai_voice_run_classifier_image_quantized();

    // close driver to reset memory, file pointer
    if (ret != VOICE_EI_IMPULSE_OK) {
        drpai_close(impulse->nn_input_frame_size);
        first_run = true;
    }
    else {
        // drpai_reset();
        first_run = false;
    }

    // default point to the output of the drpai
    float *drpai_output = drpai_output_buf;

    if (graph_config->object_detection_last_layer == VOICE_EI_CLASSIFIER_LAST_LAYER_YOLOV5_V5_DRPAI) {
#if ((VOICE_EI_CLASSIFIER_OBJECT_DETECTION == 1) && (VOICE_EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER == VOICE_EI_CLASSIFIER_LAST_LAYER_YOLOV5_V5_DRPAI))
        // do post processing
        drpai_run_yolov5_postprocessing(impulse, &drpai_output);
#endif
    }

#if VOICE_EI_LOG_LEVEL == VOICE_EI_LOG_LEVEL_DEBUG
    voice_ei_printf("First 20 bytes: ");
    for (size_t ix = 0; ix < 20; ix++) {
        voice_ei_printf("%f ", drpai_output[ix]);
    }
    voice_ei_printf("\n");
#endif

    result->_raw_outputs[learn_block_index].matrix = new matrix_t(1, graph_config->output_features_count);
    result->_raw_outputs[learn_block_index].blockId = block_config->block_id;

    for (size_t i = 0; i < graph_config->output_features_count; i++) {
        result->_raw_outputs[learn_block_index].matrix->buffer[i] = drpai_output[i];
    }

    result->timing.classification_us = voice_ei_read_timer_us() - ctx_start_us;
    return VOICE_EI_IMPULSE_OK;
}

#endif // #if (VOICE_EI_CLASSIFIER_INFERENCING_ENGINE == VOICE_EI_CLASSIFIER_DRPAI)
#endif // _VOICE_EI_CLASSIFIER_INFERENCING_ENGINE_DRPAI_H_
