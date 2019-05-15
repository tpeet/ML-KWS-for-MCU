/*
 * Copyright (C) 2019 Tanel Peet. All rights reserved.
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

#ifndef FEATURE_EXTRACTOR_FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_FEATURE_EXTRACTOR_H_
#include "arm_math.h"

#define EXTRACTION_METHOD 0
#define NUM_FBANK_BINS 10
#define LOW_FREQ 1000
#define HIGH_FREQ 8000
#define SAMP_FREQ 16000

class FeatureExtractor {

public:
	virtual ~FeatureExtractor();
	virtual void compute_features(const int16_t* audio_data, q7_t* out_data)=0;

};



#endif /* FEATURE_EXTRACTOR_FEATURE_EXTRACTOR_H_ */
