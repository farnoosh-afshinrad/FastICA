import numpy as np
from numpy.f2py.auxfuncs import throw_error
import pyroomacoustics as pra
from scipy.io import wavfile

class CombineVoices:
    def __init__(self, s1_file, s2_file, seed=0):
        # initialize RNG
        np.random.seed(seed)

        # load files
        sr1, s1 = wavfile.read(s1_file)
        sr2, s2 = wavfile.read(s2_file)
        self.room=None

        # make sure sample rates match
        if sr1 != sr2:
            raise ValueError(f"Sampling rates differ: {sr1} vs {sr2}")
        min_len = min(s1.shape[0], s2.shape[0])
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        # store as float32 arrays
        self.sr = sr1
        self.s1 = s1.astype(np.float32)
        self.s2 = s2.astype(np.float32)

        print(f"s1 shape: {self.s1.shape}, s2 shape: {self.s2.shape}")


    def save_mics(self, path='./voices/'):
        if self.room is None:
            throw_error("Room not simulated yet. Call mix_sources() first.")
        mixed_mic1=self.room.mic_array.signals[0]
        mixed_mic2=self.room.mic_array.signals[1]
        stereo = np.column_stack((mixed_mic1, mixed_mic2))
        wavfile.write(path + 'mic1.wav', self.sr, stereo)


    def mix_sources(
        self,
        mic_spacing: float = 0.05,
        room_dim = [6.0, 5.0, 3.0],
        absorption: float = 1.0,
        max_order: int = 0,
        mic_center = [3.0, 2.5, 1.7],
        apply_noise: bool = False
    ):
        """
        Simulate mixing self.s1 and self.s2 in a shoebox room,
        recorded by two mics mic_spacing apart.

        Returns:
          A 2-column numpy array of shape (n_samples, 2),
          where column 0 is mic1, column 1 is mic2.
        """
        # 1. Create shoebox room
        room = pra.ShoeBox(
            room_dim,
            fs=self.sr,
            max_order=max_order,
            absorption=absorption
        )

        # 2. Add the two sources at arbitrary distinct positions
        room.add_source([1.0, 2.0, 1.5], signal=self.s1)
        room.add_source([4.5, 3.5, 1.2], signal=self.s2)

        # 3. Build a 2-mic array, centered at mic_center, spacing=mic_spacing
        center = np.array(mic_center)
        half = mic_spacing / 2
        mic_positions = np.c_[
            center + np.array([-half, 0.0, 0.0]),
            center + np.array([ half, 0.0, 0.0])
        ]
        room.add_microphone_array(
            pra.MicrophoneArray(mic_positions, room.fs)
        )

        # 4. Compute RIRs and simulate
        room.compute_rir()
        room.simulate()

        # 5. Extract signals (shape: [2, n_samples])
        signals = room.mic_array.signals

        # 6. Optionally add Gaussian noise
        if apply_noise:
            signals = signals + 0.02 * np.random.randn(*signals.shape)

        self.room = room
