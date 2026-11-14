//! SECAM method 1 colour-under restoration. The recorder took the bell
//! pre-emphasis off the studio chroma block, passed the block through a
//! divide-by-4 counter and put a bell pre-emphasis of its own on the result,
//! so the tape holds rest carriers at foB/4 and foR/4 with the FM deviation
//! divided by four as well, under an anti-bell centred between them.
//! Playback mirrors the recorder stage by stage: the bell that undoes the
//! tape's anti-bell, a limiter, the carrier phase multiplied back up, and the
//! BT.470 anti-bell on the restored block. Carrier and deviation scale
//! together, so tape timebase error self-corrects and there is no conversion
//! LO to servo.

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use super::*;
use crate::spec::{next_fast_len, sosfiltfft};

/// Centre of the HF pre-emphasis of the chroma block (ITU-R BT.470); the
/// tape's own pre-emphasis sits at the same place counted down.
const BELL_F0_HZ: f64 = 4_286_000.0;
/// Carrier envelope, relative to the field median, the limiter starts at.
/// Below it the amplitude passes through, so a carrier that collapses
/// (dropouts, no colour) reaches the downstream colour killers as a collapse
/// rather than as full-scale noise.
const LIMITER_KNEE: f32 = 0.5;
/// First line with chroma output.
const STARTING_LINE: usize = 16;

/// HF pre-emphasis (anti-bell) network centred on `f0` at `freq_hz`:
/// (1 + j16F) / (1 + j1.26F) with F = f/f0 - f0/f (BT.470-6 table 2 item
/// 2.13). Its inverse is the bell.
fn anti_bell(freq_hz: f64, f0: f64) -> Complex64 {
    let bell_f = freq_hz / f0 - f0 / freq_hz;
    Complex64::new(1.0, 16.0 * bell_f) / Complex64::new(1.0, 1.26 * bell_f)
}

/// The playback chain over a whole field, in the spectrum of the field
/// zero-padded to a fast transform length. Both networks are applied as the
/// complex responses of the analog circuits rather than as zero-phase gains:
/// the bell has to take the phase of the recorder's anti-bell back out along
/// with its amplitude, and the anti-bell has to hand the downstream bell the
/// phase it will take out.
pub(crate) struct Method1Playback {
    r2c: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    forward: Arc<dyn Fft<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    /// Per unique bin: the bell around the under carriers, doubled into the
    /// analytic signal.
    bell: Vec<Complex32>,
    /// Per unique bin: the anti-bell within the block band, scaled to the
    /// output amplitude through the transform pair.
    anti_bell: Vec<Complex32>,
    mult: u32,
}

impl Method1Playback {
    /// For fields of up to `len` samples at `samp_rate`, of a carrier counted
    /// down by `mult`. `rest_amplitude` is the peak output amplitude of the
    /// undeviated carrier at the bell centre; `block_band` band-limits the
    /// restored block the way a forward-backward pass of it would.
    pub(crate) fn new(
        len: usize,
        samp_rate: f64,
        mult: u32,
        rest_amplitude: f64,
        block_band: &[Sos<f64>],
    ) -> Self {
        let n = next_fast_len(len);
        let bin_hz = samp_rate / n as f64;
        let under_f0 = BELL_F0_HZ / f64::from(mult);
        // Neither band reaches DC or the Nyquist bin, which the analytic
        // signal and the real inverse both want kept apart.
        let per_bin = |response: &dyn Fn(usize, f64) -> Complex64| -> Vec<Complex32> {
            (0..=n / 2)
                .map(|bin| {
                    if bin == 0 || bin == n / 2 {
                        return Complex32::new(0.0, 0.0);
                    }
                    let value = response(bin, bin as f64 * bin_hz);
                    Complex32::new(value.re as f32, value.im as f32)
                })
                .collect()
        };
        let bell = per_bin(&|_, freq| 2.0 / anti_bell(freq, under_f0));
        // A unit phasor of the positive half spectrum comes back from the
        // real inverse at 2n.
        let band = sosfiltfft(block_band, n);
        let scale = rest_amplitude / (2 * n) as f64;
        let anti_bell =
            per_bin(&|bin, freq| anti_bell(freq, BELL_F0_HZ) * (band[bin].norm_sqr() * scale));

        let mut real_planner = RealFftPlanner::<f32>::new();
        let mut planner = FftPlanner::<f32>::new();
        Self {
            r2c: real_planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
            forward: planner.plan_fft_forward(n),
            c2r: real_planner.plan_fft_inverse(n),
            bell,
            anti_bell,
            mult,
        }
    }

    /// The chroma block restored from the colour-under `chroma`.
    fn restore(&self, chroma: &[f32]) -> Vec<f32> {
        let n = self.r2c.len();
        let len = chroma.len();
        assert!(
            len <= n,
            "chroma of {len} samples does not fit the {n}-sample transform sized for the field"
        );
        let zero = Complex32::new(0.0, 0.0);
        let mut padded = vec![0.0f32; n];
        padded[..len].copy_from_slice(chroma);
        let mut half = self.r2c.make_output_vec();
        self.r2c
            .process(&mut padded, &mut half)
            .expect("r2c forward FFT failed");

        // The under carrier as an analytic signal with the tape's anti-bell
        // taken off, which leaves the constant amplitude the divider put out.
        let mut carrier: Vec<Complex32> = half
            .iter()
            .zip(&self.bell)
            .map(|(&bin, &response)| bin * response)
            .collect();
        carrier.resize(n, zero);
        self.inverse.process(&mut carrier);

        // Limiter and multiplier: the unit phasor raised to the multiplier
        // carries the phase multiplied up, with no angle ever taken.
        let envelope: Vec<f32> = carrier[..len].iter().map(|z| z.norm_sqr().sqrt()).collect();
        let knee = median_radix(&envelope) * LIMITER_KNEE;
        for (z, &amplitude) in carrier.iter_mut().zip(&envelope) {
            *z = if amplitude > 0.0 {
                (*z / amplitude).powu(self.mult) * (amplitude / knee).min(1.0)
            } else {
                zero
            };
        }
        carrier[len..].fill(zero);

        self.forward.process(&mut carrier);
        for ((bin, &restored), &response) in half.iter_mut().zip(&carrier).zip(&self.anti_bell) {
            *bin = restored * response;
        }
        self.c2r
            .process(&mut half, &mut padded)
            .expect("c2r inverse FFT failed");
        padded.truncate(len);
        padded
    }
}

/// The restored chroma block of a field, zeroed before the first line with
/// colour. No per-line AGC: the limiter sets the carrier amplitude and the
/// anti-bell shapes it, and normalizing every line to its porch level would
/// flatten the intended foR/foB rest amplitude difference.
pub(super) fn process_chroma(
    field: &DecodedField,
    playback: &Method1Playback,
    chroma: &[f32],
) -> Vec<f32> {
    let first_line = match field.burst_start_line() {
        line if line > 0 => STARTING_LINE.max(field.output_row(line as usize)),
        _ => STARTING_LINE,
    };
    let mut block = playback.restore(chroma);
    let blanked = (first_line * field.outlinelen).min(block.len());
    block[..blanked].fill(0.0);
    block
}
