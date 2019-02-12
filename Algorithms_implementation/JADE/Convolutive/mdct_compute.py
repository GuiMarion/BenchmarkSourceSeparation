import numpy as np
import scipy.sparse
import scipy.signal
import scipy.fftpack


def mdct(audio_signal, window_function):
    """
    mdct Modified discrete cosine transform (MDCT) using the fast Fourier transform (FFT)
        audio_mdct = z.mdct(audio_signal,window_function)
    Arguments:
        audio_signal: audio signal [number_samples, 0]
        window_function: window function [window_length, 0]
        audio_mdct: audio MDCT [number_frequencies, number_times]
    Example: Compute and display the MDCT as used in the AC-3 audio coding format
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt
        # Audio signal (normalized) averaged over its channels (expanded) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)
        # Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
        window_length = 512
        alpha_value = 5
        window_function = np.kaiser(int(window_length/2)+1, alpha_value*np.pi)
        window_function2 = np.cumsum(window_function[0:int(window_length/2)])
        window_function = np.sqrt(np.concatenate((window_function2, window_function2[int(window_length/2)::-1]))
                                  / np.sum(window_function))
        # MDCT
        audio_mdct = z.mdct(audio_signal, window_function)
        # MDCT displayed in dB, s, and kHz
        plt.rc('font', size=30)
        plt.imshow(20*np.log10(np.absolute(audio_mdct)), aspect='auto', cmap='jet', origin='lower')
        plt.title('MDCT (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/(window_length/2)),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, sample_rate/2+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/2*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and window length
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil(2 * number_samples / window_length) + 1)

    # Pre and post zero-padding of the signal
    audio_signal = np.pad(audio_signal,
                          (int(window_length / 2), int((number_times + 1) * window_length / 2 - number_samples)),
                          'constant', constant_values=0)

    # Initialize the MDCT
    audio_mdct = np.zeros((int(window_length / 2), number_times))

    # Pre and post-processing arrays
    preprocessing_array = np.exp(-1j * np.pi / window_length * np.arange(0, window_length))
    postprocessing_array = np.exp(-1j * np.pi / window_length * (window_length / 2 + 1)
                                  * np.arange(0.5, window_length / 2 + 0.5))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Window the signal
        sample_index = time_index * int(window_length / 2)
        audio_segment = audio_signal[sample_index:sample_index + window_length] * window_function

        # FFT of the audio segment after pre-processing
        audio_segment = np.fft.fft(audio_segment * preprocessing_array)

        # Truncate to the first half before post-processing
        audio_mdct[:, time_index] = np.real(audio_segment[0:int(window_length / 2)] * postprocessing_array)

    return audio_mdct



def imdct(audio_mdct, window_function):
    """
    imdct Inverse modified discrete cosine transform (MDCT) using the fast Fourier transform (FFT)
        audio_signal = z.imdct(audio_mdct,window_function)
    Arguments:
        audio_mdct: audio MDCT [number_frequencies, number_times]
        window_function: window function [window_length, 0]
        audio_signal: audio signal [number_samples, 0]
    Example: Verify that the MDCT is perfectly invertible
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt
        # Audio signal (normalized) averaged over its channels (expanded) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0 ** (audio_signal.itemsize * 8 - 1))
        audio_signal = np.mean(audio_signal, 1)
        # MDCT with a slope function as used in the Vorbis audio coding format
        window_length = 2048
        window_function = np.sin(np.pi / 2
                                 * np.power(np.sin(np.pi / window_length * np.arange(0.5, window_length + 0.5)), 2))
        audio_mdct = z.mdct(audio_signal, window_function)
        # Inverse MDCT and error signal
        audio_signal2 = z.imdct(audio_mdct, window_function)
        audio_signal2 = audio_signal2[0:len(audio_signal)]
        error_signal = audio_signal - audio_signal2
        # Original, resynthesized, and error signals displayed in s
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1), plt.plot(audio_signal), plt.autoscale(tight=True), plt.title("Original Signal")
        plt.xticks(np.arange(sample_rate, len(audio_signal), sample_rate),
                   np.arange(1, int(np.floor(len(audio_signal) / sample_rate)) + 1))
        plt.xlabel('Time (s)')
        plt.subplot(3, 1, 2), plt.plot(audio_signal2), plt.autoscale(tight=True), plt.title("Resynthesized Signal")
        plt.xticks(np.arange(sample_rate, len(audio_signal), sample_rate),
                   np.arange(1, int(np.floor(len(audio_signal) / sample_rate)) + 1))
        plt.xlabel('Time (s)')
        plt.subplot(3, 1, 3), plt.plot(error_signal), plt.autoscale(tight=True), plt.title("Error Signal")
        plt.xticks(np.arange(sample_rate, len(audio_signal), sample_rate),
                   np.arange(1, int(np.floor(len(audio_signal) / sample_rate)) + 1))
        plt.xlabel('Time (s)')
        plt.show()
    """

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_mdct)

    # Number of samples for the signal
    number_samples = number_frequencies * (number_times + 1)

    # Initialize the audio signal
    audio_signal = np.zeros(number_samples)

    # Pre and post-processing arrays
    preprocessing_array = np.exp(-1j * np.pi / (2 * number_frequencies)
                                 * (number_frequencies + 1) * np.arange(0, number_frequencies))
    postprocessing_array = np.exp(-1j * np.pi / (2 * number_frequencies)
                                  * np.arange(0.5 + number_frequencies / 2,
                                              2 * number_frequencies + number_frequencies / 2 + 0.5)) \
        / number_frequencies

    # FFT of the frames after pre-processing
    audio_mdct = np.fft.fft(audio_mdct.T * preprocessing_array, n=2 * number_frequencies, axis=1)

    # Apply the window to the frames after post-processing
    audio_mdct = 2 * (np.real(audio_mdct * postprocessing_array) * window_function).T

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Recover the signal thanks to the time-domain aliasing cancellation (TDAC) principle
        sample_index = time_index * number_frequencies
        audio_signal[sample_index:sample_index + 2 * number_frequencies] \
            = audio_signal[sample_index:sample_index + 2 * number_frequencies] + audio_mdct[:, time_index]

    # Remove the pre and post zero-padding
    audio_signal = audio_signal[number_frequencies:-number_frequencies - 1]

    return audio_signal