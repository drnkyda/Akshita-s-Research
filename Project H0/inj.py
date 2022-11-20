
m __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

import bilby
from bilby.core.prior import Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
%matplotlib inlineprint

(bilby.__version__)

np.random.seed(1234)

injection_parameters = dict(
            mass_1=1.5 mass_2=1. , a_1=0.05, a_2=0.05, tilt_1=0.00, tilt_2=0.0,
                phi_12=0.0, phi_jl=0.0, luminosity_distance=35., theta_jn=0.0, psi=0.0,
                    phase=0.0, geocent_time=1502973664., ra=3.45, dec= -0.4)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                  reference_frequency=50., minimum_frequency=20., catch_waveform_errors=True)
duration = 4.
sampling_frequency = 2048.

waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                        waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
                start_time=injection_parameters['geocent_time'] - 3)
injection = ifos.inject_signal(
            waveform_generator=waveform_generator,
                parameters=injection_parameters)

H1 = ifos[0]
H1_injection = injection[0]

fig, ax = plt.subplots()
idxs = H1.strain_data.frequency_mask  # This is a boolean mask of the frequencies which we'll use in the analysis
ax.loglog(H1.strain_data.frequency_array[idxs],
                  np.abs(H1.strain_data.frequency_domain_strain[idxs]),
                            label="data")
ax.loglog(H1.frequency_array[idxs],
                  H1.amplitude_spectral_density_array[idxs],
                            label="ASD")
ax.loglog(H1.frequency_array[idxs],
                  np.abs(H1_injection["plus"][idxs]),
                            label="Abs. val. of plus polarization")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Strain [strain/$\sqrt{Hz}$]")
ax.legend()
plt.show()


prior = bilby.core.prior.PriorDict()
prior['chirp_mass'] = Uniform(name='chirp_mass', minimum=0.4,maximum=4.4)
prior['mass_ratio'] = Uniform(name='mass_ratio', minimum=0.125, maximum=1)
# We fix the rest of the parameters to their injected values
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
                    'dec','luminosity_distance', 'theta_jn', 'phase', 'geocent_time']:
        prior[key] = injection_parameters[key]
prior

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, priors=prior,
                time_marginalization=False, phase_marginalization=False, distance_marginalization=False)

result_short = bilby.run_sampler(
            likelihood, prior, sampler='dynesty', outdir='short', label="GW150914",
                conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
                    nlive=50, dlogz=3,  # <- Arguments are used to make things fast - not recommended for general use
                        clean=True
                        )


result_short.posterior
result_short.posterior["chirp_mass"]
Mc = result_short.posterior["chirp_mass"].values

lower_bound = np.quantile(Mc, 0.05)
upper_bound = np.quantile(Mc, 0.95)
median = np.quantile(Mc, 0.5)
print("Mc = {} with a 90% C.I = {} -> {}".format(median, lower_bound, upper_bound))

fig, ax = plt.subplots()
ax.hist(result_short.posterior["chirp_mass"], bins=20)
ax.axvspan(lower_bound, upper_bound, color='C1', alpha=0.4)
ax.axvline(median, color='C1')
ax.set_xlabel("chirp mass")
plt.show()

result_short.plot_corner(parameters=["chirp_mass", "mass_ratio", "geocent_time"], prior=True)

parameters = dict(mass_1=1.5, mass_2=1.0)
result_short.plot_corner(parameters)

result_short.priors

result_short.sampler_kwargs["nlive"]

print("ln Bayes factor = {} +/- {}".format(
        result_short.log_bayes_factor, result_short.log_evidence_err))


