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
 */

#ifndef __KWS_GFCC_H__
#define __KWS_GFCC_H__
#include <stdbool.h>
#include <stdio.h>
#include "arm_math.h"
#include "string.h"
#include "gfcc_data.h"
#include "feature_extractor.h"

#define SAMP_FREQ 16000

#define LOW_FREQ 1000
#define HIGH_FREQ 8000

#define M_2PI 6.283185307179586476925286766559005
#define EarQ 9.26449
#define minBW 24.7
#define bw 228.832903 			// EarQ*minBW
#define sqrP 2.414213562373095 	//sqrt(3+pow(2,1.5))
#define sqrM 0.4142135623730948	//sqrt(3-pow(2,1.5))
#define order 1
#define width 1
#define GTord 4


class GFCC: public FeatureExtractor{
  private:
	int dctCoeffCount;
	int frameLen;
	int gfccDecBits;
	float32_t * _logbands;
	static float32_t const _filterCoefficients[NUM_BANDS][SPECTRUM_SIZE];
	static float32_t const _dctTable[NUM_DCT_COEFFS][NUM_BANDS];
    float32_t * frame;
    int frame_len_padded;
    float32_t * spectrum;
    float32_t * window_func;
    arm_rfft_fast_instance_f32 * rfft;
    int spectrumSize;
    int iteration;
    static float32_t const _gfccMaximums[NUM_DCT_COEFFS];
    static float32_t const _gfccMinimums[NUM_DCT_COEFFS];


    void createFilters();
//    void createDCTMatrix();

    static inline float32_t abs_complex(float32_t real, float32_t imag) {
      return sqrt(real*real + imag*imag);
    }

    static inline float32_t abs_complex(float32_t * array) {
      return sqrt(array[0]*array[0] + array[1]*array[1]);
    }


    static inline float32_t amp2db(float32_t value) {
    	if (value < 0.000000001) {
    		return -90.0;
    	}

    	return 20.0*log10(value);
    }

  public:
    GFCC(int numberBands, int filterSize, int gfcc_dec_bits);
    ~GFCC();
    void compute_features(const int16_t* data, q7_t* gfcc_out);
};

#endif
