import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

class CombineVoices:
    def __init__(self, s1_file, s2_file, seed=0):
        # initialize RNG
        np.random.seed(seed)

        # load files
        sr1, s1 = wavfile.read(s1_file)
        sr2, s2 = wavfile.read(s2_file)
        self.room = None

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
        
        # Normalize sources
        self.s1 = self.s1 / (np.max(np.abs(self.s1)) + 1e-10)
        self.s2 = self.s2 / (np.max(np.abs(self.s2)) + 1e-10)

        print(f"s1 shape: {self.s1.shape}, s2 shape: {self.s2.shape}")

    def save_mics(self, path='./voices/'):
        if self.room is None:
            raise ValueError("Room not simulated yet. Call mix_sources() first.")
        
        # Save individual microphone recordings
        for i in range(2):
            signal = self.room.mic_array.signals[i]
            # Normalize to prevent clipping
            signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.9
            filename = f'{path}mic{i+1}.wav'
            wavfile.write(filename, self.sr, signal.astype(np.float32))
            print(f"Saved: {filename}")
        
        # Also save as stereo
        mixed_mic1 = self.room.mic_array.signals[0]
        mixed_mic2 = self.room.mic_array.signals[1]
        stereo = np.column_stack((mixed_mic1, mixed_mic2))
        stereo = stereo / (np.max(np.abs(stereo)) + 1e-10) * 0.9
        wavfile.write(path + 'mixed_stereo.wav', self.sr, stereo.astype(np.float32))

    def mix_sources_instantaneous(self, mic_spacing=0.015):
        """
        Create instantaneous mixing that works with ICA
        Simulates close microphones but without time delays
        """
        print(f"\nCreating instantaneous mixing (ICA-compatible)")
        print(f"Simulating {mic_spacing*100:.1f}cm mic spacing effects")
        
        # Create mixing matrix based on mic spacing
        # Closer mics = more similar mixing coefficients
        
        if mic_spacing <= 0.02:  # Very close (phone-like)
            # Almost identical mixing, but still invertible
            A = np.array([[0.95, 0.85],
                         [0.90, 0.90]])
            noise_level = 0.01
        elif mic_spacing <= 0.05:  # Moderate
            A = np.array([[0.9, 0.6],
                         [0.7, 0.8]])
            noise_level = 0.008
        else:  # Larger spacing
            A = np.array([[0.9, 0.4],
                         [0.5, 0.85]])
            noise_level = 0.005
        
        # Add small random perturbation for realism
        A += np.random.normal(0, 0.02, A.shape)
        
        print(f"Mixing matrix A:")
        print(A)
        print(f"Condition number: {np.linalg.cond(A):.2f}")
        
        # Stack sources
        S = np.array([self.s1, self.s2])
        
        # Mix signals: X = A @ S
        X = A @ S
        
        # Add realistic noise
        X += noise_level * np.random.randn(*X.shape)
        
        # Create dummy room object for compatibility
        self.room = type('obj', (object,), {
            'mic_array': type('obj', (object,), {
                'signals': X
            })()
        })()
        
        return X
    
    def mix_sources(
        self,
        mic_spacing: float = 0.015,  # 1.5cm - typical phone mic distance
        room_dim = [6.0, 5.0, 3.0],
        absorption: float = 0.2,      # Some absorption for realism
        max_order: int = 3,           # Include some reflections
        mic_center = [3.0, 2.5, 1.7],
        apply_noise: bool = True      # Add realistic noise
    ):
        """
        Simulate mixing self.s1 and self.s2 in a shoebox room,
        recorded by two mics mic_spacing apart.
        
        Default parameters simulate phone-like recording conditions.
        """
        print(f"\nCreating room simulation with {mic_spacing*100:.1f}cm mic spacing")
        
        # 1. Create shoebox room
        room = pra.ShoeBox(
            room_dim,
            fs=self.sr,
            max_order=max_order,
            absorption=absorption
        )

        # 2. Add sources at different positions
        # Place sources at reasonable distances from mic array
        # This creates a realistic mixing scenario
        source1_pos = [2.0, 1.5, 1.7]  # Left side
        source2_pos = [4.0, 3.5, 1.7]  # Right side
        
        room.add_source(source1_pos, signal=self.s1)
        room.add_source(source2_pos, signal=self.s2)
        
        print(f"Source 1 at: {source1_pos}")
        print(f"Source 2 at: {source2_pos}")

        # 3. Build a 2-mic array with SMALL spacing (phone-like)
        center = np.array(mic_center)
        half = mic_spacing / 2
        
        # Horizontal mic array (typical phone configuration)
        mic_positions = np.c_[
            center + np.array([-half, 0.0, 0.0]),
            center + np.array([half, 0.0, 0.0])
        ]
        
        print(f"Mic array center: {mic_center}")
        print(f"Mic 1 at: {mic_positions[:, 0]}")
        print(f"Mic 2 at: {mic_positions[:, 1]}")
        
        room.add_microphone_array(
            pra.MicrophoneArray(mic_positions, room.fs)
        )

        # 4. Compute RIRs and simulate
        room.compute_rir()
        room.simulate()

        # 5. Extract signals (shape: [2, n_samples])
        signals = room.mic_array.signals

        # 6. Add realistic noise (SNR ~30-40 dB)
        if apply_noise:
            signal_power = np.mean(signals**2)
            noise_power = signal_power / (10**(35/10))  # 35 dB SNR
            noise = np.sqrt(noise_power) * np.random.randn(*signals.shape)
            room.mic_array.signals = signals + noise
            print(f"Added noise at ~35 dB SNR")

        self.room = room
        
        # Print mixing information
        print(f"\nMixing characteristics:")
        print(f"Room dimensions: {room_dim}")
        print(f"Absorption coefficient: {absorption}")
        print(f"Max reflection order: {max_order}")
        
        # Calculate and show how similar the two mic signals are
        mic1 = room.mic_array.signals[0]
        mic2 = room.mic_array.signals[1]
        correlation = np.corrcoef(mic1, mic2)[0, 1]
        print(f"Correlation between mic signals: {correlation:.4f}")
        print(f"(Higher correlation = more similar = harder to separate)")

    def mix_sources_nearfield(
        self,
        mic_spacing: float = 0.015,
        room_dim = [4.0, 3.5, 2.8],
        absorption: float = 0.6,  # Higher absorption
        max_order: int = 1,       # Only first-order reflections
    ):
        """
        Create more ICA-friendly room simulation
        - Sources closer to mics (nearfield)
        - Higher absorption (less reverb)
        - Lower reflection order
        """
        print(f"\nCreating nearfield room simulation with {mic_spacing*100:.1f}cm mic spacing")
        
        # Create room
        room = pra.ShoeBox(
            room_dim,
            fs=self.sr,
            max_order=max_order,
            absorption=absorption
        )

        # Place sources closer to microphones (nearfield configuration)
        # This reduces the effect of reflections
        mic_center = [2.0, 1.75, 1.4]
        
        # Sources at 45-degree angles, 0.5-1m from mic array
        source1_pos = [1.5, 1.25, 1.4]  # 45° left, ~0.7m away
        source2_pos = [2.5, 2.25, 1.4]  # 45° right, ~0.7m away
        
        room.add_source(source1_pos, signal=self.s1)
        room.add_source(source2_pos, signal=self.s2)
        
        print(f"Source 1 at: {source1_pos} (left)")
        print(f"Source 2 at: {source2_pos} (right)")

        # Microphone array
        center = np.array(mic_center)
        half = mic_spacing / 2
        
        mic_positions = np.c_[
            center + np.array([-half, 0.0, 0.0]),
            center + np.array([half, 0.0, 0.0])
        ]
        
        print(f"Mic array at: {mic_center}")
        print(f"Distance between sources: {np.linalg.norm(np.array(source1_pos) - np.array(source2_pos)):.2f}m")
        
        room.add_microphone_array(
            pra.MicrophoneArray(mic_positions, room.fs)
        )

        # Compute RIRs and simulate
        room.compute_rir()
        room.simulate()

        # Add slight noise
        signals = room.mic_array.signals
        signal_power = np.mean(signals**2)
        noise_power = signal_power / (10**(40/10))  # 40 dB SNR
        room.mic_array.signals = signals + np.sqrt(noise_power) * np.random.randn(*signals.shape)

        self.room = room
        
        # Report mixing quality
        mic1 = room.mic_array.signals[0]
        mic2 = room.mic_array.signals[1]
        correlation = np.corrcoef(mic1, mic2)[0, 1]
        print(f"\nMixing characteristics:")
        print(f"Absorption: {absorption} (higher = less reverb)")
        print(f"Max order: {max_order} (lower = less complex)")
        print(f"Mic correlation: {correlation:.4f}")
        
        return room.mic_array.signals

    def create_hybrid_mixing(self, mic_spacing=0.015):
        """
        Hybrid approach: Instantaneous mixing + small reverb
        This is more realistic while still being ICA-friendly
        """
        print(f"\nCreating hybrid mixing (instantaneous + light reverb)")
        
        # First create instantaneous mixing
        X_instant = self.mix_sources_instantaneous(mic_spacing)
        
        # Then add very light room effect
        # Create simple reverb using delays
        delay_samples = int(0.001 * self.sr)  # 1ms delay
        reverb_gain = 0.1  # Very light reverb
        
        X_reverb = np.zeros_like(X_instant)
        X_reverb[:, delay_samples:] = reverb_gain * X_instant[:, :-delay_samples]
        
        # Combine
        X_final = X_instant + X_reverb
        
        # Update room object
        self.room.mic_array.signals = X_final
        
        print(f"Added light reverb: {reverb_gain*100:.0f}% at {delay_samples/self.sr*1000:.1f}ms")
        
        return X_final