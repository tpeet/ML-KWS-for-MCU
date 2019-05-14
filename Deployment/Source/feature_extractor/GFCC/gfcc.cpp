/*
 * Copyright (C) Tanel Peet. All rights reserved.
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

/*
 * Description: GFCC feature extraction to match the Python implementation
 */
#include "em_chip.h"
#include "float.h"
#include "gfcc.h"

const float32_t GFCC::_dctTable[NUM_DCT_COEFFS][NUM_BANDS]=DCT_TABLE;
const float32_t GFCC::_filterCoefficients[NUM_BANDS][SPECTRUM_SIZE]=FILTER_COEFFS;
const float32_t GFCC::_gfccMinimums[NUM_DCT_COEFFS] = GFCC_MINIMUMS;
const float32_t GFCC::_gfccMaximums[NUM_DCT_COEFFS] = GFCC_MAXIMUMS;

GFCC::GFCC(int dctCoeffCount, int frameLen, int gfccDecBits)
:dctCoeffCount(dctCoeffCount),
 frameLen(frameLen),
 gfccDecBits(gfccDecBits)
{
	iteration = 0;
	printf("Initializing GFCC\r\n");
	_logbands = new float32_t[NUM_BANDS];

	// Round-up to nearest power of 2.
	frame_len_padded = pow(2,ceil((log(frameLen)/log(2))));

	frame = new float[frame_len_padded];
	spectrum = new float[frame_len_padded];
	spectrumSize = (int)((frame_len_padded/2)+1);

	printf("Creating window function\r\n");
	window_func = new float[frameLen];
	for (int i = 0; i < frameLen; i++)
		window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (frameLen));

	printf("Initializing FFT\r\n");
	// Initialize FFT
	rfft = new arm_rfft_fast_instance_f32;
	arm_rfft_fast_init_f32(rfft, frame_len_padded);

}


void GFCC::compute_features(const int16_t * audio_data, q7_t* out_data) {
	int32_t i, j;

	//  TensorFlow way of normalizing .wav data to (-1,1)
	for (i = 0; i < frameLen; i++) {
		frame[i] = (float)audio_data[i]/(1<<15);

	}

	//Fill up remaining with zeros
	memset(&frame[frameLen], 0, sizeof(float) * (frame_len_padded-frameLen));

	for (i = 0; i < frameLen; i++) {
		frame[i] *= window_func[i];
	}

	//Compute FFT
	arm_rfft_fast_f32(rfft, frame, spectrum, 0);

  //Convert to power spectrum
  //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
	int32_t half_dim = frame_len_padded/2;
	float first_energy = spectrum[0] * spectrum[0],
		last_energy =  spectrum[1] * spectrum[1];  // handle this special case
	spectrum[0] = round(first_energy*32767);
//	printf("%f  ", spectrum[0]);
	for (i = 1; i < half_dim; i++) {
		float real = spectrum[i*2], im = spectrum[i*2 + 1];
		spectrum[i] = round((real*real + im*im)*32767);
//		printf("%f  ", spectrum[i]);
	}

	spectrum[half_dim] = round(last_energy*32767);

	float32_t band;
	for (i=0; i<NUM_BANDS; ++i) {
		band = 0;
		for (j=0; j<spectrumSize; ++j) {
			band += (spectrum[j] * spectrum[j]) * _filterCoefficients[i][j];
		}
		_logbands[i] = amp2db(band);
	}

	float32_t sum;
	for (i=0; i<dctCoeffCount; ++i) {
		sum = 0.0;
		for (int j=0; j<NUM_BANDS; ++j) {
			sum += _logbands[j] * _dctTable[i][j];
		}

		sum = 2*(sum-_gfccMinimums[i])/(_gfccMaximums[i]-_gfccMinimums[i]) - 1;

	    sum *= (0x1<<gfccDecBits);
	    sum = round(sum);
	    if(sum >= 127)
	    	out_data[i] = 127;
	    else if(sum <= -128)
	    	out_data[i] = -128;
	    else
	    	out_data[i] = sum;
	}
	iteration++;
}

GFCC::~GFCC() {
}
