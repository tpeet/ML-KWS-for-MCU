import math
import cmath
import numpy as np


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
