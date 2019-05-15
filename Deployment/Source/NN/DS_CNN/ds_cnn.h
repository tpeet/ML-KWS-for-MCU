/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 *
 * Modifications Copyright 2019 Tanel Peet. All rights reserved.
 * Renamed variables for using multiple feature extraction methods. Code generated
 * automatically
 */

#ifndef __DS_CNN_H__
#define __DS_CNN_H__

#include "nn.h"
#include "ds_cnn_weights.h"
#include "local_NN.h"
#include "arm_math.h"

#define SAMP_FREQ 16000
#define FEATURES_DEC_BITS 5
#define FRAME_SHIFT_MS 32
#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))
#define NUM_FRAMES 61
#define NUM_FEATURES_COEFFS 5
#define FRAME_LEN_MS 64
#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))

#define IN_DIM (NUM_FRAMES*NUM_FEATURES_COEFFS)
#define OUT_DIM 2

#define CONV1_OUT_CH 32
#define CONV1_IN_X NUM_FEATURES_COEFFS
#define CONV1_IN_Y NUM_FRAMES
#define CONV1_KX 5
#define CONV1_KY 5
#define CONV1_SX 2
#define CONV1_SY 2
#define CONV1_PX 2
#define CONV1_PY 2
#define CONV1_OUT_X 3
#define CONV1_OUT_Y 31

#define CONV2_OUT_CH 32
#define CONV2_IN_X CONV1_OUT_X
#define CONV2_IN_Y CONV1_OUT_Y
#define CONV2_DS_KX 3
#define CONV2_DS_KY 3
#define CONV2_DS_SX 1
#define CONV2_DS_SY 1
#define CONV2_DS_PX 1
#define CONV2_DS_PY 1
#define CONV2_OUT_X 3
#define CONV2_OUT_Y 31

#define CONV3_OUT_CH 32
#define CONV3_IN_X CONV2_OUT_X
#define CONV3_IN_Y CONV2_OUT_Y
#define CONV3_DS_KX 3
#define CONV3_DS_KY 3
#define CONV3_DS_SX 1
#define CONV3_DS_SY 1
#define CONV3_DS_PX 1
#define CONV3_DS_PY 1
#define CONV3_OUT_X 3
#define CONV3_OUT_Y 31

#define CONV4_OUT_CH 32
#define CONV4_IN_X CONV3_OUT_X
#define CONV4_IN_Y CONV3_OUT_Y
#define CONV4_DS_KX 3
#define CONV4_DS_KY 3
#define CONV4_DS_SX 1
#define CONV4_DS_SY 1
#define CONV4_DS_PX 1
#define CONV4_DS_PY 1
#define CONV4_OUT_X 3
#define CONV4_OUT_Y 31

#define CONV1_BIAS_LSHIFT 4
#define CONV1_OUT_RSHIFT 5
#define CONV2_DS_BIAS_LSHIFT 4
#define CONV2_DS_OUT_RSHIFT 7
#define CONV2_PW_BIAS_LSHIFT 4
#define CONV2_PW_OUT_RSHIFT 8
#define CONV3_DS_BIAS_LSHIFT 2
#define CONV3_DS_OUT_RSHIFT 4
#define CONV3_PW_BIAS_LSHIFT 5
#define CONV3_PW_OUT_RSHIFT 7
#define CONV4_DS_BIAS_LSHIFT 4
#define CONV4_DS_OUT_RSHIFT 6
#define CONV4_PW_BIAS_LSHIFT 5
#define CONV4_PW_OUT_RSHIFT 8
#define FINAL_FC_BIAS_LSHIFT 3
#define FINAL_FC_OUT_RSHIFT 9
#define AVG_POOL_OUT_LSHIFT 4

#define SCRATCH_BUFFER_SIZE (2*2*CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY + 2*CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y)

class DS_CNN : public NN {

  public:
    DS_CNN();
    ~DS_CNN();
    void run_nn(q7_t* in_data, q7_t* out_data);

  private:
    q7_t* scratch_pad;
    q7_t* col_buffer;
    q7_t* buffer1;
    q7_t* buffer2;
    static q7_t const conv1_wt[CONV1_OUT_CH*CONV1_KX*CONV1_KY];
    static q7_t const conv1_bias[CONV1_OUT_CH];
    static q7_t const conv2_ds_wt[CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY];
    static q7_t const conv2_ds_bias[CONV1_OUT_CH];
    static q7_t const conv2_pw_wt[CONV2_OUT_CH*CONV1_OUT_CH];
    static q7_t const conv2_pw_bias[CONV2_OUT_CH];
    static q7_t const conv3_ds_wt[CONV2_OUT_CH*CONV3_DS_KX*CONV3_DS_KY];
    static q7_t const conv3_ds_bias[CONV2_OUT_CH];
    static q7_t const conv3_pw_wt[CONV3_OUT_CH*CONV2_OUT_CH];
    static q7_t const conv3_pw_bias[CONV3_OUT_CH];
    static q7_t const conv4_ds_wt[CONV3_OUT_CH*CONV4_DS_KX*CONV4_DS_KY];
    static q7_t const conv4_ds_bias[CONV3_OUT_CH];
    static q7_t const conv4_pw_wt[CONV4_OUT_CH*CONV3_OUT_CH];
    static q7_t const conv4_pw_bias[CONV4_OUT_CH];
    static q7_t const final_fc_wt[CONV4_OUT_CH*OUT_DIM];
    static q7_t const final_fc_bias[OUT_DIM];

};

#endif
