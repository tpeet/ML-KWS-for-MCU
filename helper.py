import math
import cmath
import numpy as np



class ParameterExtractor(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def get_param(self, param):
        if param in self.parameters:
            return self.parameters[param]
        if param in self.parameters['search_space']:
            return self.parameters['search_space'][param]
        return None

    def get_model_size_info(self):
        # set parameters
        msi = self.get_param('model_size_info')
        num_layers = msi['num_layers']
        model_size_info = [num_layers]
        for i in range(num_layers):
            layer = msi['layers'][i]
            model_size_info.append(int(layer['num_channels']))
            if 'sx' in layer.keys():
                model_size_info.append(int(layer['sx']))
            else:
                model_size_info.append(3)

            if 'sy' in layer.keys():
                model_size_info.append(int(layer['sx']))
            else:
                model_size_info.append(3)

            if i == 0:
                model_size_info.append(2)
                model_size_info.append(2)
            else:
                model_size_info.append(1)
                model_size_info.append(1)
        return model_size_info



def find_parameter(parameters, parameter):
    if parameter in parameters:
        return parameters[parameter]
    if parameter in parameters['search_space']:
        return parameters['search_space'][parameter]
    return None


def write_filter_coeffs(fname, inputSize, sampleRate, numberBands, maxFrequency, minFrequency, width=1):
    EarQ = 9.26449
    minBW = 24.7

    filterSize = numberBands
    filterSizeInv = 1.0 / filterSize
    bw = EarQ * minBW
    filterFrequencies = [0] * filterSize
    for i in range(1, filterSize + 1):
        filterFrequencies[filterSize - i] = -bw + math.exp(
            i * (-1 * math.log(maxFrequency + bw) + math.log(minFrequency + bw)) * filterSizeInv) * (maxFrequency + bw)

    filterCoefficients = []
    order = 1
    fftSize = (inputSize - 1) * 2;
    oneJ = complex(0, 1)
    ucirc = []
    for i in range(inputSize):
        ucirc.append(cmath.exp((oneJ * 2.0 * math.pi * i) / fftSize))

    sqrP = math.sqrt(3 + pow(2, 1.5))
    sqrM = math.sqrt(3 - pow(2, 1.5))
    with open(fname, 'a') as f:
        f.write("#define FILTER_COEFFS {")
        for i in range(filterSize):
            cf = filterFrequencies[i]
            ERB = width * pow((pow((cf / EarQ), order) + pow(minBW, order)), 1.0 / order)
            B = 1.019 * 2 * math.pi * ERB
            r = math.exp(-B / sampleRate)
            theta = 2 * math.pi * cf / sampleRate
            pole = r * cmath.exp(oneJ * theta)
            T = 1.0 / sampleRate
            GTord = 4

            sinCf = math.sin(2 * cf * math.pi * T)
            cosCf = math.cos(2 * cf * math.pi * T);
            gtCos = 2 * T * cosCf / math.exp(B * T);
            gtSin = T * sinCf / math.exp(B * T);

            A11 = -(gtCos + 2 * sqrP * gtSin) / 2
            A12 = -(gtCos - 2 * sqrP * gtSin) / 2
            A13 = -(gtCos + 2 * sqrM * gtSin) / 2
            A14 = -(gtCos - 2 * sqrM * gtSin) / 2

            zeros = [-A11 / T, -A12 / T, -A13 / T, -A14 / T]

            g1 = -2 * cmath.exp(4 * oneJ * cf * math.pi * T) * T
            g2 = 2 * cmath.exp(-(B * T) + 2 * oneJ * cf * math.pi * T) * T
            cxExp = cmath.exp(4 * oneJ * cf * math.pi * T)

            filterGain = abs(
                (g1 + g2 * (cosCf - sqrM * sinCf)) *
                (g1 + g2 * (cosCf + sqrM * sinCf)) *
                (g1 + g2 * (cosCf - sqrP * sinCf)) *
                (g1 + g2 * (cosCf + sqrP * sinCf)) /
                pow((-2 / cmath.exp(2 * B * T) - 2 * cxExp + 2 * (1 + cxExp) / math.exp(B * T)), 4))

            filterCoeffs = []
            f.write("{")
            for j in range(inputSize):
                filterCoeffsElement = (pow(T, 4) / filterGain) * abs(ucirc[j] - zeros[0]) * abs(
                    ucirc[j] - zeros[1]) * abs(ucirc[j] - zeros[2]) * abs(ucirc[j] - zeros[3]) * pow(
                    abs((pole - ucirc[j]) * (pole - ucirc[j])), (-GTord))
                filterCoeffs.append(filterCoeffsElement)

                if j == (inputSize - 1):
                    f.write('{}'.format(filterCoeffsElement))
                else:
                    f.write('{}, '.format(filterCoeffsElement))

            filterCoefficients.append(filterCoeffs)
            if i == (filterSize - 1):
                f.write("}")
            else:
                f.write("}, ")
        f.write("}\n")


def write_dct_table(fname, inputSize, outputSize):
    # void DCT::createDctTableII(int inputSize, int outputSize)
    scale0 = 1.0 / math.sqrt(inputSize)
    scale1 = math.sqrt(2.0 / inputSize)
    dctTable = []
    with open(fname, 'a') as f:
        f.write("#define DCT_TABLE {")
        for i in range(outputSize):
            scale = scale1
            if i == 0:
                scale = scale0
            freqMultiplier = math.pi / inputSize * i

            dctRow = []
            f.write("{")
            for j in range(inputSize):
                dctElement = scale * math.cos(freqMultiplier * (j + 0.5))
                dctRow.append(dctElement)
                if j == (inputSize - 1):
                    f.write("{}".format(dctElement))
                else:
                    f.write("{}, ".format(dctElement))
            dctTable.append(dctRow)
            if i == (outputSize - 1):
                f.write("}")
            else:
                f.write("}, ")

        f.write("} \n")


def write_gfcc_tables(fname, parameters, gfcc_minimums, gfcc_maximums):
    number_fbank_filters = find_parameter(parameters, 'num_fbank_filters')
    sample_rate = find_parameter(parameters, 'sample_rate')
    samples_in_frame = int(find_parameter(parameters, 'window_size_ms') * sample_rate / 1000)
    spectrum_length = int(samples_in_frame / 2) + 1
    dct_coeff_count = int(find_parameter(parameters, 'dct_coefficient_count'))
    low_frequency_bound = int(find_parameter(parameters, 'lower_frequency'))
    high_frequency_bound = int(find_parameter(parameters, 'upper_frequency'))

    with open(fname, 'w') as f:
        f.write('#define NUM_BANDS {}\n'.format(number_fbank_filters))
        f.write('#define SPECTRUM_SIZE {}\n'.format(spectrum_length))
        f.write('#define NUM_DCT_COEFFS {}\n'.format(dct_coeff_count))
        f.write('#define GFCC_MINIMUMS {')
        np.array(np.array(gfcc_minimums[:dct_coeff_count]).tofile(f, sep=", ", format="%f"))
        f.write('} \n')
        f.write('#define GFCC_MAXIMUMS {')
        np.array(np.array(gfcc_maximums[:dct_coeff_count]).tofile(f, sep=", ", format="%f"))
        f.write('} \n')

    write_filter_coeffs(fname, samples_in_frame, sample_rate, number_fbank_filters, high_frequency_bound,
                        low_frequency_bound)
    write_dct_table(fname, number_fbank_filters, dct_coeff_count)




def write_ds_cnn_h_beginning(f, wanted_words, sample_rate, clip_duration_ms,
                             window_size_ms, window_stride_ms, dct_coefficient_count,
                             model_size_info, act_max):
    f.write("#ifndef __DS_CNN_H__\n")
    f.write("#define __DS_CNN_H__\n\n")
    f.write('#include "nn.h"\n')
    f.write('#include "ds_cnn_weights.h"\n')
    f.write('#include "local_NN.h"\n')
    f.write('#include "arm_math.h"\n\n')
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    input_x = dct_coefficient_count
    input_y = spectrogram_length

    f.write("#define SAMP_FREQ {}\n".format(sample_rate))
    f.write("#define MFCC_DEC_BITS {}\n".format(int(7 - np.log2(act_max[0]))))
    f.write("#define FRAME_SHIFT_MS {}\n".format(int(window_stride_ms)))
    f.write("#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))\n")
    f.write("#define NUM_FRAMES {}\n".format(spectrogram_length))
    f.write("#define NUM_MFCC_COEFFS {}\n".format(dct_coefficient_count))
    f.write("#define FRAME_LEN_MS {}\n".format(int(window_size_ms)))
    f.write("#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))\n\n")

    f.write("#define IN_DIM (NUM_FRAMES*NUM_MFCC_COEFFS)\n")
    f.write("#define OUT_DIM {}\n\n".format(int(len(wanted_words.split(',')) + 1)))

    num_layers = model_size_info[0]
    i = 1
    for layer_no in range(1, num_layers + 1):
        f.write("#define CONV{}_OUT_CH {}\n".format(layer_no, model_size_info[i]))
        i += 1
        ky = model_size_info[i]
        i += 1
        kx = model_size_info[i]
        i += 1
        sy = model_size_info[i]
        i += 1
        sx = model_size_info[i]
        out_x = math.ceil(float(input_x) / float(sx))
        out_y = math.ceil(float(input_y) / float(sy))
        pad_x = max((out_x - 1) * sx + kx - input_x, 0) // 2
        pad_y = max((out_y - 1) * sy + ky - input_y, 0) // 2
        if layer_no == 1:
            f.write("#define CONV1_IN_X NUM_MFCC_COEFFS\n")
            f.write("#define CONV1_IN_Y NUM_FRAMES\n")
            f.write("#define CONV{}_KX {}\n".format(layer_no, kx))
            f.write("#define CONV{}_KY {}\n".format(layer_no, ky))
            f.write("#define CONV{}_SX {}\n".format(layer_no, sx))
            f.write("#define CONV{}_SY {}\n".format(layer_no, sy))
            f.write("#define CONV{}_PX {}\n".format(layer_no, pad_x))
            f.write("#define CONV{}_PY {}\n".format(layer_no, pad_y))
            f.write("#define CONV{}_OUT_X {}\n".format(layer_no, out_x))
            f.write("#define CONV{}_OUT_Y {}\n".format(layer_no, out_y))


        else:
            f.write("#define CONV{1}_IN_X CONV{0}_OUT_X\n".format(layer_no - 1, layer_no))
            f.write("#define CONV{1}_IN_Y CONV{0}_OUT_Y\n".format(layer_no - 1, layer_no))
            f.write("#define CONV{}_DS_KX {}\n".format(layer_no, kx))
            f.write("#define CONV{}_DS_KY {}\n".format(layer_no, ky))
            f.write("#define CONV{}_DS_SX {}\n".format(layer_no, sx))
            f.write("#define CONV{}_DS_SY {}\n".format(layer_no, sy))
            f.write("#define CONV{}_DS_PX {}\n".format(layer_no, pad_x))
            f.write("#define CONV{}_DS_PY {}\n".format(layer_no, pad_y))
            f.write("#define CONV{0}_OUT_X {1}\n".format(layer_no, out_x))
            f.write("#define CONV{0}_OUT_Y {1}\n".format(layer_no, out_y))

        i += 1
        f.write("\n")
        input_x = out_x
        input_y = out_y


def write_ds_cnn_h_end(f, num_layers):
    f.write(
        '#define SCRATCH_BUFFER_SIZE (2*2*CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY + 2*CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y)\n\n')
    f.write('class DS_CNN : public NN {\n\n')
    f.write('  public:\n')
    f.write('    DS_CNN();\n')
    f.write('    ~DS_CNN();\n')
    f.write('    void run_nn(q7_t* in_data, q7_t* out_data);\n\n')
    f.write('  private:\n')
    f.write('    q7_t* scratch_pad;\n')
    f.write('    q7_t* col_buffer;\n')
    f.write('    q7_t* buffer1;\n')
    f.write('    q7_t* buffer2;\n')
    f.write('    static q7_t const conv1_wt[CONV1_OUT_CH*CONV1_KX*CONV1_KY];\n')
    f.write('    static q7_t const conv1_bias[CONV1_OUT_CH];\n')

    for layer_no in range(2, num_layers + 1):
        f.write(
            '    static q7_t const conv{1}_ds_wt[CONV{0}_OUT_CH*CONV{1}_DS_KX*CONV{1}_DS_KY];\n'.format(layer_no - 1,
                                                                                                        layer_no))
        f.write('    static q7_t const conv{1}_ds_bias[CONV{0}_OUT_CH];\n'.format(layer_no - 1, layer_no))
        f.write('    static q7_t const conv{1}_pw_wt[CONV{1}_OUT_CH*CONV{0}_OUT_CH];\n'.format(layer_no - 1, layer_no))
        f.write('    static q7_t const conv{0}_pw_bias[CONV{0}_OUT_CH];\n'.format(layer_no))

    f.write('    static q7_t const final_fc_wt[CONV{}_OUT_CH*OUT_DIM];\n'.format(num_layers))
    f.write('    static q7_t const final_fc_bias[OUT_DIM];\n\n')
    f.write('};\n\n')
    f.write('#endif  \n')


def write_ds_cnn_cpp_file(fname, num_layers):
    f = open(fname, 'wb')
    f.close()
    with open(fname, 'a') as f:
        f.write('#include "ds_cnn.h"\n\n')

        for layer_no in range(0, num_layers):
            if layer_no == 0:
                f.write("const q7_t DS_CNN::conv1_wt[CONV1_OUT_CH*CONV1_KX*CONV1_KY]=CONV1_WT;\n")
                f.write("const q7_t DS_CNN::conv1_bias[CONV1_OUT_CH]=CONV1_BIAS;\n")
            else:
                f.write(
                    "const q7_t DS_CNN::conv{1}_ds_wt[CONV{0}_OUT_CH*CONV{1}_DS_KX*CONV{1}_DS_KY]=CONV{1}_DS_WT;\n".format(
                        layer_no, layer_no + 1))
                f.write("const q7_t DS_CNN::conv{1}_ds_bias[CONV{0}_OUT_CH]=CONV{1}_DS_BIAS;\n".format(layer_no,
                                                                                                       layer_no + 1))
                f.write(
                    "const q7_t DS_CNN::conv{1}_pw_wt[CONV{1}_OUT_CH*CONV{0}_OUT_CH]=CONV{1}_PW_WT;\n".format(layer_no,
                                                                                                              layer_no + 1))
                f.write("const q7_t DS_CNN::conv{0}_pw_bias[CONV{0}_OUT_CH]=CONV{0}_PW_BIAS;\n".format(layer_no + 1))

        f.write("const q7_t DS_CNN::final_fc_wt[CONV{0}_OUT_CH*OUT_DIM]=FINAL_FC_WT;\n".format(num_layers))
        f.write("const q7_t DS_CNN::final_fc_bias[OUT_DIM]=FINAL_FC_BIAS;\n\n")

        f.write("DS_CNN::DS_CNN()\n")
        f.write("{\n")
        f.write("    scratch_pad = new q7_t[SCRATCH_BUFFER_SIZE];\n")
        f.write("    buffer1 = scratch_pad;\n")
        f.write("    buffer2 = buffer1 + (CONV1_OUT_CH*CONV1_OUT_X*CONV1_OUT_Y);\n")
        f.write("    col_buffer = buffer2 + (CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y);\n")
        f.write("    frame_len = FRAME_LEN;\n")
        f.write("    frame_shift = FRAME_SHIFT;\n")
        f.write("    num_mfcc_features = NUM_MFCC_COEFFS;\n")
        f.write("    num_frames = NUM_FRAMES;\n")
        f.write("    num_out_classes = OUT_DIM;\n")
        f.write("    in_dec_bits = MFCC_DEC_BITS;\n")
        f.write("}\n\n")

        f.write("DS_CNN::~DS_CNN()\n")
        f.write("{\n")
        f.write("    delete scratch_pad;\n")
        f.write("}\n\n")

        f.write("void DS_CNN::run_nn(q7_t* in_data, q7_t* out_data)\n")
        f.write("{\n")
        for layer_no in range(0, num_layers):
            if layer_no == 0:
                f.write("    //CONV1 : regular convolution\n")
                f.write(
                    "    arm_convolve_HWC_q7_basic_nonsquare(in_data, CONV1_IN_X, CONV1_IN_Y, 1, conv1_wt, CONV1_OUT_CH, CONV1_KX, CONV1_KY, CONV1_PX, CONV1_PY, CONV1_SX, CONV1_SY, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_X, CONV1_OUT_Y, (q15_t*)col_buffer, NULL);\n")
                f.write("    arm_relu_q7(buffer1,CONV1_OUT_X*CONV1_OUT_Y*CONV1_OUT_CH);\n\n\n")
            else:
                f.write("    //CONV{} : DS + PW conv\n".format(layer_no + 1))
                f.write("    //Depthwise separable conv (batch norm params folded into conv wts/bias)\n")
                f.write(
                    "    arm_depthwise_separable_conv_HWC_q7_nonsquare(buffer1,CONV{1}_IN_X,CONV{1}_IN_Y,CONV{0}_OUT_CH,conv{1}_ds_wt,CONV{0}_OUT_CH,CONV{1}_DS_KX,CONV{1}_DS_KY,CONV{1}_DS_PX,CONV{1}_DS_PY,CONV{1}_DS_SX,CONV{1}_DS_SY,conv{1}_ds_bias,CONV{1}_DS_BIAS_LSHIFT,CONV{1}_DS_OUT_RSHIFT,buffer2,CONV{1}_OUT_X,CONV{1}_OUT_Y,(q15_t*)col_buffer, NULL);\n".format(
                        layer_no, layer_no + 1))
                f.write("    arm_relu_q7(buffer2,CONV{0}_OUT_X*CONV{0}_OUT_Y*CONV{0}_OUT_CH);\n\n".format(layer_no + 1))

                f.write("    //Pointwise conv\n")
                f.write(
                    "    arm_convolve_1x1_HWC_q7_fast_nonsquare(buffer2, CONV{1}_OUT_X, CONV{1}_OUT_Y, CONV{0}_OUT_CH, conv{1}_pw_wt, CONV{1}_OUT_CH, 1, 1, 0, 0, 1, 1, conv{1}_pw_bias, CONV{1}_PW_BIAS_LSHIFT, CONV{1}_PW_OUT_RSHIFT, buffer1, CONV{1}_OUT_X, CONV{1}_OUT_Y, (q15_t*)col_buffer, NULL);\n".format(
                        layer_no, layer_no + 1))
                f.write(
                    "    arm_relu_q7(buffer1,CONV{0}_OUT_X*CONV{0}_OUT_Y*CONV{0}_OUT_CH);\n\n\n".format(layer_no + 1))

        f.write("    //Average pool\n")
        f.write(
            "    arm_avepool_q7_HWC_nonsquare (buffer1,CONV{0}_OUT_X,CONV{0}_OUT_Y,CONV{0}_OUT_CH,CONV{0}_OUT_X,CONV{0}_OUT_Y,0,0,1,1,1,1,NULL,buffer2, AVG_POOL_OUT_LSHIFT);\n".format(
                num_layers))
        f.write(
            "    arm_fully_connected_q7(buffer2, final_fc_wt, CONV{0}_OUT_CH, OUT_DIM, FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT, final_fc_bias, out_data, (q15_t*)col_buffer);\n".format(
                num_layers))
        f.write("}\n")

